import sys
import os
import rclpy
import signal
import time
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import zarr
import cv2
from pynput import keyboard

# 导入全局配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class DiffusionPolicyCollector(Node):
    def __init__(self):
        super().__init__('diffusion_policy_collector')
        
        # 1. 直接引用配置
        self.dt = config['control_period']   
        self.lidar_dim = config['lidar_dim']
        self.save_dir = config['dataset_path']
        self.is_exiting = False

        # 速度发布者
        self.cmd_pub = self.create_publisher(Twist, config["topic_cmd_vel"], 10)

        # 2. 键盘监听：直接通过 config 中的 range 边界进行赋值
        self.curr_cmd = np.array([0.0, 0.0], dtype=np.float32)
        def on_press(key):
            try:
                k = key.char.lower()
                # 严格匹配 config 里的键名：v_max, v_min, w_max, w_min
                if k == 'w':   self.curr_cmd = np.array([config['action_stats']['v_max'], 0.0], dtype=np.float32)
                elif k == 's': self.curr_cmd = np.array([config['action_stats']['v_min'], 0.0], dtype=np.float32)
                elif k == 'a': self.curr_cmd = np.array([0.0, config['action_stats']['w_max']], dtype=np.float32)
                elif k == 'd': self.curr_cmd = np.array([0.0, config['action_stats']['w_min']], dtype=np.float32)
            except AttributeError:
                if key == keyboard.Key.space:
                    self.curr_cmd = np.array([0.0, 0.0], dtype=np.float32)

        self.kb_listener = keyboard.Listener(on_press=on_press)
        self.kb_listener.start()
        
        os.makedirs(self.save_dir, exist_ok=True)
            
        # 3. 内存缓存区
        self.images, self.states, self.actions = [], [], []
        self.curr_img = None
        self.curr_scan = None
        
        # 4. 全量订阅 (不管训练用不用图，采集时一定全录)
        self.get_logger().info(f">>> 采集节点就绪 | 路径: {self.save_dir}")
        self.create_subscription(Image, config["topic_img"], self.img_cb, 10)
        self.create_subscription(LaserScan, config["topic_lidar"], self.scan_cb, 10)

        self.record_timer = self.create_timer(self.dt, self.record_tick)

    def img_cb(self, msg): self.curr_img = msg
    def scan_cb(self, msg): self.curr_scan = msg

    def record_tick(self):
        if self.is_exiting: return

        # A. 物理驱动
        twist = Twist()
        twist.linear.x, twist.angular.z = float(self.curr_cmd[0]), float(self.curr_cmd[1])
        self.cmd_pub.publish(twist)
        
        # B. 数据同步录入
        try:
            # 必须保证 Image 和 Lidar 都有数据才开始记录第一帧
            if self.curr_img is not None and self.curr_scan is not None:
                # 图像解析
                img_data = np.frombuffer(self.curr_img.data, dtype=np.uint8)
                if self.curr_img.encoding == 'rgb8':
                    img_np = img_data.reshape(self.curr_img.height, self.curr_img.width, 3)
                else:
                    yuyv_img = img_data.reshape(self.curr_img.height, self.curr_img.width, 2)
                    img_np = cv2.cvtColor(yuyv_img, cv2.COLOR_YUV2RGB_YUYV)
                
                # 雷达重采样
                ranges = np.nan_to_num(np.array(self.curr_scan.ranges), nan=0.0, 
                                      posinf=self.curr_scan.range_max, neginf=0.0)
                scan_res = np.interp(np.linspace(0, len(ranges)-1, self.lidar_dim), 
                                     np.arange(len(ranges)), ranges).astype(np.float32)

                self.images.append(img_np)
                self.states.append(scan_res)
                self.actions.append(self.curr_cmd.copy())

                sys.stdout.write(f"\r[Recording] Frames: {len(self.actions)} | V: {self.curr_cmd[0]:.2f} W: {self.curr_cmd[1]:.2f}")
                sys.stdout.flush()
        except Exception as e:
            self.get_logger().error(f"Sampling Error: {e}")

    def save_zarr(self):
        if self.is_exiting or len(self.actions) < 10:
            print("\n[Skip] Data too short.")
            return
        
        self.is_exiting = True
        print("\n\n>>> Saving Zarr Data... <<<")
        
        try:
            idx = 0
            while os.path.exists(os.path.join(self.save_dir, f"episode_{idx}.zarr")): idx += 1
            save_path = os.path.join(self.save_dir, f"episode_{idx}.zarr")

            root = zarr.open(save_path, mode='w')
            data_group = root.create_group('data')
            
            data_group.create_dataset('img', data=np.stack(self.images), chunks=(100, *self.images[0].shape), dtype='uint8')
            data_group.create_dataset('state', data=np.stack(self.states), chunks=(100, self.lidar_dim), dtype='float32')
            data_group.create_dataset('action', data=np.stack(self.actions), chunks=(100, 2), dtype='float32')
            
            meta_group = root.create_group('meta')
            meta_group.create_dataset('episode_ends', data=np.array([len(self.actions)], dtype=np.int64))
            
            print(f"✅ Success: {save_path}")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    def signal_handler(sig, frame): pass
    signal.signal(signal.SIGINT, signal_handler)
    
    rclpy.init()
    node = DiffusionPolicyCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_zarr()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()