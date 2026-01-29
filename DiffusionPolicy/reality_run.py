import sys
import os
import signal
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import torch

# 路径处理
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from train_diffusion_policy.model import DiffusionTransformer
from train_diffusion_policy.diffusion import Diffusion
from train_diffusion_policy.inference import InferenceRunner

class DiffusionPolicyROS(Node):
    def __init__(self):
        super().__init__("diffusion_policy_inference_node")

        # --- 1. 参数对接 ---
        self.device = torch.device(config["device"])
        self.lidar_dim = config["lidar_dim"]
        self.control_dt = config['control_period'] 
        
        # --- 2. 模型核心组件加载 ---
        self.model = DiffusionTransformer().to(self.device)
        
        if os.path.exists(config["model_path"]):
            state_dict = torch.load(config["model_path"], map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.get_logger().info(f"✅ 权重加载成功: {config['model_path']}")
        else:
            self.get_logger().error(f"❌ 找不到模型权重: {config['model_path']}")
            return

        self.model.eval()
        self.diff = Diffusion().to(self.device)
        
        # 推理器：它内部会维护 obs_history (滑动窗口)
        self.runner = InferenceRunner(self.model, self.diff)

        # --- 3. 最新数据暂存 (由回调更新) ---
        self.current_scan = None
        self.current_image = None

        # --- 4. ROS 通信 ---
        if config["use_lidar"]:
            self.create_subscription(LaserScan, config["topic_lidar"], self.scan_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, config["topic_cmd_vel"], 10)

        # --- 5. 推理时钟 ---
        self.timer = self.create_timer(self.control_dt, self.control_loop)
        
        self.get_logger().info("-" * 40)
        self.get_logger().info(f"🚀 部署节点启动 | 频率: {1/self.control_dt:.1f}Hz")
        self.get_logger().info("-" * 40)

    def scan_callback(self, msg):
        """仅负责数据预处理和暂存"""
        ranges = np.array(msg.ranges)
        # 基础清理
        ranges = np.nan_to_num(ranges, nan=config['lidar_max'], posinf=config['lidar_max'], neginf=config['lidar_min'])
        
        # 维度对齐
        if len(ranges) != self.lidar_dim:
            self.current_scan = np.interp(
                np.linspace(0, len(ranges) - 1, self.lidar_dim),
                np.arange(len(ranges)),
                ranges
            ).astype(np.float32)
        else:
            self.current_scan = ranges.astype(np.float32)

    def control_loop(self):
        """主控制循环 - 统一数据流"""
        # 1. 等待第一帧数据到达
        if self.current_scan is None:
            return

        # 2. 构造观测字典 (传单帧数据给 runner)
        # 注意：不要在这里加维度，InferenceRunner 内部会处理 [1, T, D]
        obs_raw = {
            "state": self.current_scan, 
            "img": self.current_image # 目前为 None
        }

        try:
            start_time = time.time()
            
            # 3. 执行推理 (内部完成归一化、历史堆叠、去噪)
            # 返回值已由 InferenceRunner 反归一化为物理值 [v, w]
            action = self.runner.predict_action(obs_raw)
            
            # 4. 获取速度指令
            v_raw, w_raw = action[0], action[1]

            # 5. 二次安全限幅 (双重保险)
            v_cmd = np.clip(float(v_raw), config['action_stats']['v_min'], config['action_stats']['v_max'])
            w_cmd = np.clip(float(w_raw), config['action_stats']['w_min'], config['action_stats']['w_max'])

            # 6. 发布
            cmd_msg = Twist()
            cmd_msg.linear.x = v_cmd
            cmd_msg.angular.z = w_cmd
            self.cmd_pub.publish(cmd_msg)

            duration = time.time() - start_time
            # 降低日志频率，避免阻塞终端
            if self.get_clock().now().to_msg().nanosec % 5 == 0:
                self.get_logger().info(f"✔ 推理成功 | v: {v_cmd:.2f}, w: {w_cmd:.2f} | 耗时: {duration:.3f}s")

        except Exception as e:
            import traceback
            self.get_logger().error(f"❌ 推理异常: {str(e)}\n{traceback.format_exc()}")

def main():
    rclpy.init()
    node = DiffusionPolicyROS()
    
    # 捕获 Ctrl+C
    def stop_and_exit(sig, frame):
        stop_msg = Twist()
        # 发布 0 速度防止机器人乱跑
        node.cmd_pub.publish(stop_msg)
        node.get_logger().info("🛑 紧急制动并安全退出")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_and_exit)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()