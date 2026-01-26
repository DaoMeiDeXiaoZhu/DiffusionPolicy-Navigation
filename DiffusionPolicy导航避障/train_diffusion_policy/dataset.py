import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zarr
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from config import config


class ZarrDataset(Dataset):
    def __init__(self, root_dir=None, horizon=None):
        """
        root_dir: 数据集路径（默认从 config 加载）
        horizon:  序列长度（默认从 config 加载）
        """

        # 加载配置
        self.root_dir = root_dir if root_dir is not None else config["dataset_path"]
        self.horizon = horizon if horizon is not None else config["horizon"]

        # 收集所有 episode 的路径
        self.episodes = []
        for name in os.listdir(self.root_dir):
            if name.endswith(".zarr"):
                self.episodes.append(os.path.join(self.root_dir, name))

        # 生成所有可采样的 (episode_id, start_index)
        self.indices = []
        for epi_id, path in enumerate(self.episodes):
            root = zarr.open(path, mode='r')
            T = root["data"]["img"].shape[0]

            for t in range(T - self.horizon):
                self.indices.append((epi_id, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        epi_id, t = self.indices[idx]
        root = zarr.open(self.episodes[epi_id], mode='r')

        # 读取图像、雷达、动作序列
        imgs = root["data"]["img"][t:t+self.horizon]
        states = root["data"]["state"][t:t+self.horizon]
        actions = root["data"]["action"][t:t+self.horizon]

        # 图像处理：归一化并调整维度
        imgs = torch.tensor(imgs, dtype=torch.float32) / 255.0
        imgs = imgs.permute(0, 3, 1, 2)

        # 雷达与动作处理
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        obs = {
            "img": imgs,
            "state": states
        }

        return obs, actions
