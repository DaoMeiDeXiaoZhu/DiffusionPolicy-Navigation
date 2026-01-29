import sys
import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset

# 确保导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class ZarrDataset(Dataset):
    def __init__(self, root_dir=None, horizon=None):
        self.root_dir = root_dir if root_dir is not None else config["dataset_path"]
        self.horizon = horizon if horizon is not None else config["horizon"]
        # 【修改点】统一使用 obs_steps，且直接从 config 读取，不再作为 init 参数硬编码
        self.obs_steps = config["obs_steps"] 
        
        self.episode_data = []
        self.indices = []
        
        zarr_paths = sorted([os.path.join(self.root_dir, d) for d in os.listdir(self.root_dir) if d.endswith('.zarr')])
        
        print(f"正在加载数据集，obs_steps={self.obs_steps}, horizon={self.horizon}")
        
        for epi_id, path in enumerate(zarr_paths):
            root = zarr.open(path, mode='r')
            self.episode_data.append(root)
            
            T = root["data"]["action"].shape[0]
            # 合法的起始点：确保从 t 开始有连续 horizon 步动作
            for t in range(T - self.horizon + 1):
                self.indices.append((epi_id, t))
        
        print(f"数据集加载完成，总采样序列数: {len(self.indices)}")

    def normalize_action(self, action):
        """根据 config 进行 Min-Max 归一化到 [-1, 1]"""
        v = action[..., 0]
        w = action[..., 1]
        v_norm = 2 * (v - config['action_stats']['v_min']) / (config['action_stats']['v_max'] - config['action_stats']['v_min']) - 1
        w_norm = 2 * (w - config['action_stats']['w_min']) / (config['action_stats']['w_max'] - config['action_stats']['w_min']) - 1
        return torch.stack([v_norm, w_norm], dim=-1)

    def normalize_lidar(self, lidar):
        """雷达数据归一化到 [0, 1]"""
        lidar = torch.clamp(lidar, config['lidar_min'], config['lidar_max'])
        lidar_norm = (lidar - config['lidar_min']) / (config['lidar_max'] - config['lidar_min'])
        return lidar_norm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        epi_id, t = self.indices[idx]
        root = self.episode_data[epi_id]

        # --- 1. 处理观察值 (Observation) ---
        # 【修改点】使用 self.obs_steps。取 [t - obs_steps + 1 : t + 1]
        start_obs = t - self.obs_steps + 1
        end_obs = t + 1
        
        # 处理边界情况：如果起始点不足以提供历史帧，则用第一帧重复填充 (Padding)
        if start_obs < 0:
            pad_len = abs(start_obs)
            raw_states = root["data"]["state"][0 : end_obs]
            states = np.concatenate([np.repeat(raw_states[:1], pad_len, axis=0), raw_states], axis=0)
            
            if config['use_img']:
                raw_imgs = root["data"]["img"][0 : end_obs]
                imgs = np.concatenate([np.repeat(raw_imgs[:1], pad_len, axis=0), raw_imgs], axis=0)
        else:
            states = root["data"]["state"][start_obs : end_obs]
            if config['use_img']:
                imgs = root["data"]["img"][start_obs : end_obs]

        # --- 2. 处理动作值 (Action Horizon) ---
        # 动作预测从 t 开始的未来 horizon 步
        actions = root["data"]["action"][t : t + self.horizon]

        # --- 3. 转换为 Tensor 并归一化 ---
        states = self.normalize_lidar(torch.from_numpy(states).float())
        actions = self.normalize_action(torch.from_numpy(actions).float())

        obs = {"state": states} # 形状 (obs_steps, lidar_dim)
        
        if config['use_img']:
            imgs = torch.from_numpy(imgs).float() / 255.0
            # 形状 (obs_steps, C, H, W)
            obs["img"] = imgs.permute(0, 3, 1, 2) 

        return obs, actions