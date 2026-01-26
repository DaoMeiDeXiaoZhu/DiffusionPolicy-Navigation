import os
import torch

base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "collect_expert_experience/expert_data_collection")
model_path = os.path.join(base_path, "latest_dp_model.pth")

config = {
    # ============================================================
    # 训练超参数
    # ============================================================
    'learning_rate': 1e-4,     # Adam 学习率
    'batchsize': 2,            # DataLoader batch size
    'epochs': 30,              # 训练轮数
    'max_norm': 1.0,           # 梯度裁剪

    # ============================================================
    # Diffusion Policy 参数
    # ============================================================
    'horizon': 8,              # 动作序列长度 T
    'timesteps': 1000,         # 扩散步数
    'beta_start': 1e-4,        # β 起始值
    'beta_end': 0.02,          # β 结束值

    # ============================================================
    # 模型结构参数
    # ============================================================
    'state_dim': 360,          # 雷达维度
    'action_dim': 2,           # 动作维度 (v, w)
    'embed_dim': 64,           # Transformer embedding 维度
    'hidden_dim': 128,         # Transformer FFN 隐藏层
    'n_layers': 3,             # Transformer 层数

    # ============================================================
    # 图像相关
    # ============================================================
    'image_scale': 1/4,        # 图像缩放比例（480×640 → 120×160）
    'raw_img_height': 480,     # 图像原始高度
    'raw_img_width': 640,      # 图像原始宽度
    'use_img': False,          # 是否使用图像
    'use_lidar': True,         # 是否使用激光雷达

    # ============================================================
    # ROS2 话题配置
    # ============================================================
    'topic_img': "/ascamera/camera_publisher/rgb0/image",
    'topic_lidar': "/scan_raw",
    'topic_cmd_vel': "/cmd_vel",

    # ============================================================
    # 设备
    # ============================================================
    'device': "cuda" if torch.cuda.is_available() else "cpu",

    # ============================================================
    # 路径
    # ============================================================
    'dataset_path': dataset_path,
    'model_path': model_path,
}
