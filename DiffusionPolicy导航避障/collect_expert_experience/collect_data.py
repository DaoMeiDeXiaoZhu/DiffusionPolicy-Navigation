import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import zarr
import os
import sys
import cv2
import subprocess
import signal
import shutil # 用于文件操作

class DiffusionPolicyCollector(Node):
    def __init__(self):
        super().__init__('diffusion_policy_collector')
        
        # ========== 键盘控制部分保持不变 ==========
        self.teleop_process = None
        self.is_exiting = False
        try:
            self.teleop_process = subprocess.Popen(
                ["ros2", "run", "teleop_twist_keyboard", "teleop_twist_keyboard"],
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            self.get_logger().info(f">>> 键盘控制节点已启动 (PID: {self.teleop_process.pid}) <<<")
        except Exception as e:
            self.get_logger().error(f"启动键盘控制节点失败：{str(e)}")
            sys.exit(1)
        
        # 保存路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, 'expert_data_collection')
        os.makedirs(self.save_dir, exist_ok=True)
            
        # 内存缓存区
        self.images = []
        self.states = [] # 改名：scans -> states (作为低维向量输入)
        self.actions = []
        
        self.curr_img = None
        self.curr_scan = None
        self.curr_cmd = np.array([0.0, 0.0], dtype=np.float32)
        
        # 订阅器
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.img_cb, 10)
        self.create_subscription(LaserScan, '/scan_raw', self.scan_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)

        # 10Hz采集
        self.record_timer = self.create_timer(0.1, self.record_tick)

        self.get_logger().info(f">>> 采集节点已就绪 <<<")
        self.get_logger().info(f"保存格式: Zarr (符合 Diffusion Policy ReplayBuffer)")

    def img_cb(self, msg):
        self.curr_img = msg

    def scan_cb(self, msg):
        self.curr_scan = msg

    def cmd_cb(self, msg):
        self.curr_cmd = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def record_tick(self):
        if self.is_exiting or self.curr_img is None or self.curr_scan is None:
            return

        try:
            # --- 图像处理 (保持你的逻辑) ---
            img_h = self.curr_img.height
            img_w = self.curr_img.width
            img_data = np.frombuffer(self.curr_img.data, dtype=np.uint8)
            
            # 这里省略了你原有的详细编码检查代码，假设已经转为 img_np (RGB)
            # 实际运行时请保留你原有的 YUYV/RGB 转换逻辑
            # 为演示简洁，这里直接用 cv2 转换作为示例，请替换回你原来的完整逻辑
            if self.curr_img.encoding == 'rgb8':
                img_np = img_data.reshape(img_h, img_w, 3)
            else:
                 # 简化的回退逻辑，请保留你原代码中完善的 YUYV 处理
                yuyv_img = img_data.reshape(img_h, img_w, 2)
                img_np = cv2.cvtColor(yuyv_img, cv2.COLOR_YUV2RGB_YUYV)

            # --- 雷达处理 ---
            # 同样保持你的重采样逻辑
            ranges = np.nan_to_num(np.array(self.curr_scan.ranges), nan=0.0, 
                                  posinf=self.curr_scan.range_max, neginf=0.0)
            scan_resampled = np.interp(np.linspace(0, len(ranges)-1, 360), 
                                      np.arange(len(ranges)), ranges).astype(np.float32)

            action = self.curr_cmd.copy()

            # --- 存入 Buffer ---
            self.images.append(img_np) # uint8, (H, W, 3)
            self.states.append(scan_resampled) # float32, (360,)
            self.actions.append(action) # float32, (2,)

            sys.stdout.write(f"\r[Recording] {len(self.images):<6} 帧 | v: {action[0]:.2f} w: {action[1]:.2f}")
            sys.stdout.flush()
            
        except Exception as e:
            pass

    def save_zarr(self):
        """符合官方 ReplayBuffer 格式的保存逻辑"""
        if self.is_exiting:
            return
        
        self.is_exiting = True
        print("\n\n>>> 正在按 Diffusion Policy 格式保存数据... <<<")
        
        if len(self.images) < 10:
            print("[警告] 数据过少，不保存。")
        else:
            try:
                # 1. 准备数据堆叠
                # img: (T, H, W, C) - uint8 节省空间
                img_stack = np.stack(self.images, axis=0) 
                # state: (T, D) - float32
                state_stack = np.stack(self.states, axis=0)
                # action: (T, D) - float32
                action_stack = np.stack(self.actions, axis=0)
                
                # 2. 确定保存路径
                idx = 0
                while os.path.exists(os.path.join(self.save_dir, f"episode_{idx}.zarr")):
                    idx += 1
                save_path = os.path.join(self.save_dir, f"episode_{idx}.zarr")

                # 3. 创建 Zarr 根组
                # mode='w' 会覆盖同名文件夹，但我们上面已经做了重名检查
                root = zarr.open(save_path, mode='w')
                
                # 4. 创建 'data' 组 (扁平化结构)
                data_group = root.create_group('data')
                
                # --- 关键修改：直接在 data 下创建数组，不嵌套 obs ---
                
                # 图像数据：通常命名为 'img' 或 'image'
                # chunks=(100, ...) 这里的 100 是时间维度的块大小，优化读取速度
                data_group.create_dataset(
                    'img', 
                    data=img_stack, 
                    chunks=(100, *img_stack.shape[1:]), 
                    dtype='uint8' # 存为 uint8，训练时由 Dataset 类归一化
                )
                
                # 状态数据：雷达数据作为 state
                data_group.create_dataset(
                    'state', 
                    data=state_stack, 
                    chunks=(100, 360), 
                    dtype='float32'
                )
                
                # 动作数据
                data_group.create_dataset(
                    'action', 
                    data=action_stack, 
                    chunks=(100, 2), 
                    dtype='float32'
                )
                
                # 5. 创建 'meta' 组
                meta_group = root.create_group('meta')
                # episode_ends: 记录每个回合的结束索引
                # 因为这里只存了一个 episode，所以结束索引就是总长度
                meta_group.create_dataset(
                    'episode_ends', 
                    data=np.array([len(img_stack)], dtype=np.int64)
                )
                
                print(f"[成功] 数据已保存: {save_path}")
                print(f"       结构: data/img    {img_stack.shape} (uint8)")
                print(f"             data/state  {state_stack.shape} (float32)")
                print(f"             data/action {action_stack.shape} (float32)")
                print(f"             meta/episode_ends")

            except Exception as e:
                print(f"[失败] 保存出错：{str(e)}")

        # ========== 终止子进程 ==========
        if self.teleop_process:
            try:
                self.teleop_process.send_signal(signal.SIGINT)
                self.teleop_process.wait(timeout=2)
            except:
                self.teleop_process.kill()

def main():
    # 信号处理保持不变
    def signal_handler(sig, frame):
        return
    signal.signal(signal.SIGINT, signal_handler)
    
    rclpy.init(args=None)
    node = DiffusionPolicyCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_zarr()
    except Exception as e:
        print(f"Error: {e}")
        node.save_zarr()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()