import os
import torch

# ============================================================
# 0. 全局不可变参数 (修改此处将同步所有模块)
# ============================================================
GLOBAL_SETTINGS = {
    # 物理维度与时序
    'obs_steps': 4,            # 观测序列长度（过去obs_steps帧数预测未来horizon帧）
    'horizon': 12,              # 序列长度 T
    'lidar_dim': 360,          # 雷达线数
    'action_dim': 2,           # 动作维度 (v, w)
    
    # 传感器开关
    'use_img': False, 
    'use_lidar': True,
    
    # ROS 通信话题
    'topic_img': "/ascamera/camera_publisher/rgb0/image",
    'topic_lidar': "/scan_raw",
    'topic_cmd_vel': "/cmd_vel",
    
    # 时间步长控制 (10Hz)，两者保持一致
    'hz': 10,
    'dt': 0.1,                 # 1/hz

    # ============================================================
    # 数据归一化配置 (Normalization)
    # ============================================================
    # 雷达归一化范围 (米)
    'lidar_range_min': 0.05,
    'lidar_range_max': 12.0,
    
    # 动作归一化范围 (必须与你采集数据时的真实物理限速一致)
    # 建议设置为你键盘控制时的最大速度值
    'v_range_min': -0.5,       # 线速度最小值
    'v_range_max': 0.5,        # 线速度最大值
    'w_range_min': -1.5,       # 角速度最小值
    'w_range_max': 1.5,        # 角速度最大值
}

base_path = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 数据采集配置 (引用全局参数)
# ============================================================
collection_config = {
    'save_path': os.path.join(base_path, "collect_expert_experience/expert_data_collection"),
    'collect_hz': GLOBAL_SETTINGS['hz'],
    'horizon': GLOBAL_SETTINGS['horizon'],
    'topic_img': GLOBAL_SETTINGS['topic_img'],
    'topic_lidar': GLOBAL_SETTINGS['topic_lidar'],
    'topic_cmd_vel': GLOBAL_SETTINGS['topic_cmd_vel'],
}

# ============================================================
# 2. 训练配置 (引用全局参数)
# ============================================================
train_config = {
    'dataset_path': collection_config['save_path'],
    'model_save_path': os.path.join(base_path, "latest_dp_model.pth"),
    
    # 核心超参数
    'learning_rate': 3e-5,
    'batchsize': 32,
    'epochs': 300,
    'max_norm': 1.0,
    
    # Diffusion 过程
    'obs_steps': GLOBAL_SETTINGS['obs_steps'],
    'horizon': GLOBAL_SETTINGS['horizon'],
    'timesteps': 1000,
    'beta_start': 1e-4,
    'beta_end': 0.02,

    # 模型结构参数
    'lidar_dim': GLOBAL_SETTINGS['lidar_dim'],
    'action_dim': GLOBAL_SETTINGS['action_dim'],
    'embed_dim': 512,
    'hidden_dim': 1024,
    'n_layers': 8,

    # 传感器开关与预处理
    'use_img': GLOBAL_SETTINGS['use_img'],
    'use_lidar': GLOBAL_SETTINGS['use_lidar'],
    'image_scale': 1/4,
    'raw_img_height': 480,
    'raw_img_width': 640,
    
    # 硬件环境
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,

    # 训练用归一化参数
    'lidar_min': GLOBAL_SETTINGS['lidar_range_min'],
    'lidar_max': GLOBAL_SETTINGS['lidar_range_max'],
    'action_stats': {
        'v_min': GLOBAL_SETTINGS['v_range_min'],
        'v_max': GLOBAL_SETTINGS['v_range_max'],
        'w_min': GLOBAL_SETTINGS['w_range_min'],
        'w_max': GLOBAL_SETTINGS['w_range_max']
    }
}

# ============================================================
# 3. 真机部署配置 (引用全局参数)
# ============================================================
deploy_config = {
    'model_path': train_config['model_save_path'],
    'device': train_config['device'],
    'inference_steps': 50,     # DDIM 采样步数
    
    # 物理维度同步
    'obs_steps': GLOBAL_SETTINGS['obs_steps'],
    'horizon': GLOBAL_SETTINGS['horizon'],
    'lidar_dim': GLOBAL_SETTINGS['lidar_dim'],
    'action_dim': GLOBAL_SETTINGS['action_dim'],
    
    # 传感器与话题同步
    'use_img': GLOBAL_SETTINGS['use_img'],
    'use_lidar': GLOBAL_SETTINGS['use_lidar'],
    'topic_img': GLOBAL_SETTINGS['topic_img'],
    'topic_lidar': GLOBAL_SETTINGS['topic_lidar'],
    'topic_cmd_vel': GLOBAL_SETTINGS['topic_cmd_vel'],
    
    # 控制频率与归一化同步
    'control_period': GLOBAL_SETTINGS['dt'],
    'lidar_min': GLOBAL_SETTINGS['lidar_range_min'],
    'lidar_max': GLOBAL_SETTINGS['lidar_range_max'],
    'action_stats': train_config['action_stats'],
}

# 导出合并字典
config = {**train_config, **deploy_config}