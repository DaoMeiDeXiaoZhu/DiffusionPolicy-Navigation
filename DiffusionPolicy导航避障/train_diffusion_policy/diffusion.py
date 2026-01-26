import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import config


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 从配置中加载扩散参数
        timesteps = config["timesteps"]
        beta_start = config["beta_start"]
        beta_end = config["beta_end"]

        # 构造线性增长的 beta_t（噪声强度）
        betas = torch.linspace(beta_start, beta_end, timesteps)

        # 计算 α_t = 1 - β_t
        alphas = 1.0 - betas

        # 计算 ᾱ_t = ∏_{i=1..t} α_i（累计乘积）
        alpha_bar = torch.cumprod(alphas, dim=0)

        # 注册 buffer（随模型移动到 GPU，但不参与训练）
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # 扩散总步数
        self.T = timesteps

    # 前向扩散：q(x_t | x_0)
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        a_bar = self.alpha_bar[t].reshape(-1, 1, 1)

        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

        return x_t, noise

    # 反向扩散：p(x_{t-1} | x_t)
    def p_sample(self, model, obs, x_t, t):

        # 预测噪声 εθ(x_t, o, t)
        pred_noise = model(
            img=obs["img"],
            state=obs["state"],
            noisy_action=x_t,
            t=t
        )

        beta_t = self.betas[t].reshape(-1, 1, 1)
        alpha_t = self.alphas[t].reshape(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1)

        # 反向扩散均值项
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
        )

        # t = 0 时不再加噪声
        if (t == 0).all():
            return mean

        # 加入随机噪声项
        noise = torch.randn_like(x_t)

        return mean + torch.sqrt(beta_t) * noise
