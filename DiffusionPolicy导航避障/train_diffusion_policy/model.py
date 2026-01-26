import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class DiffusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # 从 config 加载模型结构参数
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        embed_dim = config["embed_dim"]
        hidden_dim = config["hidden_dim"]
        n_layers = config["n_layers"]

        # 计算缩放后的图像大小，避免显存爆炸
        img_h = int(config["raw_img_height"] * config['image_scale'])
        img_w = int(config["raw_img_width"] * config['image_scale'])

        # 图像缩放层（降低分辨率以减少显存）
        self.resize = nn.Upsample(size=(img_h, img_w), mode="bilinear")

        # CNN 图像编码器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
        )

        # 自动计算 CNN flatten 后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_h, img_w)
            out = self.cnn(dummy)
            self.cnn_out_dim = out.numel()

        # 将 CNN 输出映射到 Transformer 的 embed_dim
        self.img_fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 雷达状态编码器
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 动作编码器
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 时间步编码
        self.t_embed = nn.Embedding(config["timesteps"], embed_dim)

        # Transformer 编码器
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # 输出层：预测噪声 εθ
        self.out = nn.Linear(embed_dim, action_dim)

    def forward(self, img, state, noisy_action, t,
                use_img=None, use_lidar=None):
        """
        use_img / use_lidar 默认从 config 加载
        """
        if use_img is None:
            use_img = config["use_img"]
        if use_lidar is None:
            use_lidar = config["use_lidar"]

        B, T, C, H, W = img.shape

        # 图像编码（可选）
        if use_img:
            img_flat = img.reshape(B * T, C, H, W)
            img_flat = self.resize(img_flat)
            feat = self.cnn(img_flat)
            feat = feat.reshape(B * T, -1)
            img_emb = self.img_fc(feat).reshape(B, T, -1)
        else:
            img_emb = 0

        # 雷达编码（可选）
        if use_lidar:
            state_emb = self.state_fc(state)
        else:
            state_emb = 0

        # 动作编码
        act_emb = self.action_fc(noisy_action)

        # 时间步编码
        t_emb = self.t_embed(t).unsqueeze(1).repeat(1, T, 1)

        # 融合所有模态
        x = img_emb + state_emb + act_emb + t_emb

        # Transformer 编码
        x = self.transformer(x)

        # 输出噪声预测 εθ
        return self.out(x)
