import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import torch
from config import config
from train_diffusion_policy.model import DiffusionTransformer
from train_diffusion_policy.diffusion import Diffusion
from train_diffusion_policy.inference import InferenceRunner


class DiffusionPolicyROS(Node):
    def __init__(self):
        super().__init__("diffusion_policy_inference")

        # 加载模型路径（从 config）
        model_path = config["model_path"]

        # 创建模型结构
        self.model = DiffusionTransformer()

        # 加载模型参数
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        # 设置 eval 模式
        self.model.eval()

        # Diffusion 对象
        self.diff = Diffusion()

        # 推理器（device 从 config 加载）
        self.runner = InferenceRunner(self.model, self.diff)

        self.get_logger().info("Diffusion Policy 模型已加载完毕")

        # 观测模态选择（从 config 加载）
        self.use_img = config["use_img"]
        self.use_lidar = config["use_lidar"]

        # 图像尺寸（从 config 加载）
        self.img_h = config["img_height"]
        self.img_w = config["img_width"]

        # 订阅传感器
        self.curr_img = None
        self.curr_scan = None

        self.create_subscription(Image, config["topic_img"], self.img_cb, 10)
        self.create_subscription(LaserScan, config["topic_lidar"], self.scan_cb, 10)

        # 发布动作
        self.cmd_pub = self.create_publisher(Twist, config["topic_cmd_vel"], 10)

        # 10Hz 推理
        self.timer = self.create_timer(0.1, self.control_tick)

    # 图像回调
    def img_cb(self, msg):
        img_h = msg.height
        img_w = msg.width
        img_data = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.encoding == "rgb8":
            img_np = img_data.reshape(img_h, img_w, 3)
        else:
            yuyv_img = img_data.reshape(img_h, img_w, 2)
            img_np = cv2.cvtColor(yuyv_img, cv2.COLOR_YUV2RGB_YUYV)

        self.curr_img = img_np

    # 雷达回调
    def scan_cb(self, msg):
        ranges = np.nan_to_num(
            np.array(msg.ranges),
            nan=0.0,
            posinf=msg.range_max,
            neginf=0.0
        )
        scan_resampled = np.interp(
            np.linspace(0, len(ranges)-1, 360),
            np.arange(len(ranges)),
            ranges
        ).astype(np.float32)

        self.curr_scan = scan_resampled

    # 推理控制循环
    def control_tick(self):
        if self.use_img and self.curr_img is None:
            return
        if self.use_lidar and self.curr_scan is None:
            return

        # 图像处理（如果启用）
        if self.use_img:
            img_resized = cv2.resize(self.curr_img, (self.img_w, self.img_h))
            img = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = img.unsqueeze(0).unsqueeze(0)
        else:
            img = torch.zeros(1, 1, 3, self.img_h, self.img_w, dtype=torch.float32)

        # 雷达处理（如果启用）
        if self.use_lidar:
            state = torch.tensor(self.curr_scan, dtype=torch.float32)
            state = state.unsqueeze(0).unsqueeze(0)
        else:
            state = torch.zeros(1, 1, 360, dtype=torch.float32)

        obs = {"img": img, "state": state}

        # Diffusion Policy 推理
        action = self.runner.predict_action(obs)

        v, w = float(action[0]), float(action[1])

        # 发布 /cmd_vel
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(f"[DP] v={v:.2f}, w={w:.2f}")


def main():
    rclpy.init()
    node = DiffusionPolicyROS()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("检测到 Ctrl+C，准备停止机器人")
    finally:
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0

        for _ in range(5):
            node.cmd_pub.publish(stop)
            time.sleep(0.05)

        node.get_logger().info("已发送停止指令 /cmd_vel = 0")
        node.destroy_node()
        rclpy.shutdown()
