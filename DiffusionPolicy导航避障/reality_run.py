import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import torch
import os

from train_diffusion_policy.model import DiffusionTransformer
from train_diffusion_policy.diffusion import Diffusion
from train_diffusion_policy.inference import InferenceRunner


class DiffusionPolicyROS(Node):
    def __init__(self):
        super().__init__("diffusion_policy_inference")

        # ========== 加载模型 ==========
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(base_dir, "./train_diffusion_policy/latest_dp_model.pth")
        model_path = os.path.join(base_dir, "latest_dp_model.pth")

        self.model = DiffusionTransformer()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.diff = Diffusion()
        self.runner = InferenceRunner(self.model, self.diff, device="cpu")

        self.get_logger().info("Diffusion Policy 模型已加载完毕")

        # ========== 订阅传感器 ==========
        self.curr_img = None
        self.curr_scan = None

        self.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", self.img_cb, 10)
        self.create_subscription(LaserScan, "/scan_raw", self.scan_cb, 10)

        # ========== 发布动作 ==========
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # 10Hz 推理
        self.timer = self.create_timer(0.1, self.control_tick)

    # ---------------- 图像回调 ----------------
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

    # ---------------- 雷达回调 ----------------
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

    # ---------------- 推理控制循环 ----------------
    def control_tick(self):
        if self.curr_img is None or self.curr_scan is None:
            return

        # ========== 1. 图像缩放到训练时的尺寸 (160×120) ==========
        img_resized = cv2.resize(self.curr_img, (160, 120))  # (W,H)

        img = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,C,H,W)

        # ========== 2. 雷达 ==========
        state = torch.tensor(self.curr_scan, dtype=torch.float32)
        state = state.unsqueeze(0).unsqueeze(0)  # (1,1,360)

        obs = {
            "img": img,
            "state": state
        }

        # ========== 3. Diffusion Policy 推理 ==========
        action = self.runner.predict_action(obs)  # numpy, shape (2,)

        v, w = float(action[0]), float(action[1])

        # ========== 4. 发布 /cmd_vel ==========
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(f"[DP] v={v:.2f}, w={w:.2f}")


def main():
    rclpy.init()
    node = DiffusionPolicyROS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main():
    rclpy.init()
    node = DiffusionPolicyROS()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("检测到 Ctrl+C，准备停止机器人")

    finally:
        # ====== 退出前发送 0 速度 ======
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0

        # 连续发几次，确保底盘收到
        for _ in range(5):
            node.cmd_pub.publish(stop)
            time.sleep(0.05)

        node.get_logger().info("已发送停止指令 /cmd_vel = 0")

        node.destroy_node()
        rclpy.shutdown()

