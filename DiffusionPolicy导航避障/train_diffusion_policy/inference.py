import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import config


class InferenceRunner:
    def __init__(self, model, diffusion):
        # 推理设备（CPU 或 GPU）
        self.device = config["device"]

        # 噪声预测网络 εθ
        self.model = model.to(self.device)

        # Diffusion 对象（包含 p_sample 反向扩散）
        self.diff = diffusion

        # 动作序列长度（必须与训练一致）
        self.T = config["horizon"]

    @torch.no_grad()
    def predict_action(self, obs):
        """
        obs: {
            "img":   (1, T, C, H, W)
            "state": (1, T, 360)
        }
        返回：当前时刻的动作 (v, w)
        """

        # 初始化 x_T 为标准正态噪声（反向扩散起点）
        x = torch.randn(1, self.T, config["action_dim"], device=self.device)

        # 从 t = T-1 逐步去噪到 t = 0
        for t in reversed(range(self.diff.T)):
            tt = torch.tensor([t], device=self.device)

            x = self.diff.p_sample(
                self.model,
                obs,
                x,
                tt
            )

        # 返回最终动作序列 x_0 的第 0 帧动作
        return x[0, 0].cpu().numpy()
