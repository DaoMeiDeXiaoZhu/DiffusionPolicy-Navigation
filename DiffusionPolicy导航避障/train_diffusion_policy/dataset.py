import zarr
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class ZarrDataset(Dataset):
    def __init__(self, root_dir, horizon=16):
        # 每个样本的序列长度（Diffusion Policy 的时间窗口）
        self.horizon = horizon

        # 存放所有 episode 的路径
        self.episodes = []

        # 遍历 root_dir 下所有文件，找到所有 .zarr episode
        for name in os.listdir(root_dir):
            if name.endswith(".zarr"):
                # 保存完整路径，例如 ".../episode_0.zarr"
                self.episodes.append(os.path.join(root_dir, name))

        # 存放所有可采样的 (episode_id, start_index)
        # 例如 (0, 10) 表示第 0 个 episode 从第 10 帧开始取 horizon 长度的序列
        self.indices = []

        # 遍历每个 episode
        for epi_id, path in enumerate(self.episodes):
            # 打开 zarr 文件
            root = zarr.open(path, mode='r')

            # 当前 episode 的总帧数 T
            T = root["data"]["img"].shape[0]

            # 为该 episode 生成所有合法的起点 t
            # 例如 T=100, horizon=16 → t ∈ [0, 84]
            for t in range(T - horizon):
                self.indices.append((epi_id, t))

    def __len__(self):
        # 返回所有 episode 的所有可采样窗口数量
        return len(self.indices)

    def __getitem__(self, idx):
        # 根据 idx 找到对应的 episode 和起点 t
        epi_id, t = self.indices[idx]

        # 打开对应 episode 的 zarr 文件
        root = zarr.open(self.episodes[epi_id], mode='r')

        # 取 horizon 长度的图像序列 (T, H, W, 3)
        imgs = root["data"]["img"][t:t+self.horizon]

        # 取 horizon 长度的雷达序列 (T, 360)
        states = root["data"]["state"][t:t+self.horizon]

        # 取 horizon 长度的动作序列 (T, 2)
        actions = root["data"]["action"][t:t+self.horizon]

        # ========== 图像处理 ==========
        # 转为 float32 并归一化到 [0,1]
        imgs = torch.tensor(imgs, dtype=torch.float32) / 255.0

        # 将图像从 (T, H, W, C) 转为 (T, C, H, W)，符合 PyTorch 格式
        imgs = imgs.permute(0, 3, 1, 2)

        # ========== 雷达与动作 ==========
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        # 打包 observation
        obs = {
            "img": imgs,      # (T, C, H, W)
            "state": states   # (T, 360)
        }

        # 返回 obs 和对应的动作序列
        return obs, actions
