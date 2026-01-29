import sys
import os
import torch
import torch.nn as nn

# 确保路径正确以加载 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 从 config 加载扩散核心参数
        self.T = config["timesteps"]
        beta_start = config["beta_start"]
        beta_end = config["beta_end"]

        # 2. 构造噪声调度 (Noise Schedule)
        # 使用线性调度 (Linear Schedule)，适合大多数低维动作控制任务
        betas = torch.linspace(beta_start, beta_end, self.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 3. 注册 Buffer (随模型移动到 GPU, 不参与梯度更新)
        # 使用 register_buffer 确保这些常量在模型保存和加载时能正确同步
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def _extract(self, a, t, x_shape):
        """辅助函数：提取指定 t 的系数并适配 Tensor 形状，用于 Batch 运算"""
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    # ============================================================
    # 训练用：前向加噪 (q_sample)
    # 对应公式：x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    # ============================================================
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    # ============================================================
    # 调试/标准用：DDPM 反向采样 (p_sample)
    # ============================================================
    @torch.no_grad()
    def p_sample(self, model, obs, x_t, t):
        """
        obs['state'] 必须包含 config['obs_steps'] 帧历史数据
        x_t 必须包含 config['horizon'] 步动作序列
        """
        # 预测噪声：model 内部会将 1+obs_steps+horizon 个 Token 拼接处理
        pred_noise = model(img=obs.get("img"), state=obs["state"], noisy_action=x_t, t=t)

        # 提取当前步的系数
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # 计算去噪均值
        model_mean = sqrt_recip_alphas_t * (
            x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_noise
        )

        if (t == 0).all():
            return model_mean
        else:
            # 重构方差通常取 beta_t 或固定常数
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(beta_t) * noise

    # ============================================================
    # 部署用：DDIM 加速采样 (ddim_sample)
    # DDIM 采样允许在 50 步内获得 1000 步的效果，且采样过程是确定性的 (eta=0)
    # ============================================================
    @torch.no_grad()
    def ddim_sample(self, model, obs, x_t, t, t_next, eta=0.0):
        """
        model: DiffusionTransformer 实例
        obs: 包含 'state' (B, obs_steps, D) 和 'img' 的字典
        x_t: 当前动作轨迹 (B, horizon, action_dim)
        t: 当前时间步 Tensor
        t_next: 下一个时间步 Tensor (通常 t_next = t - (1000/steps))
        """
        # 1. 预测噪声
        pred_noise = model(img=obs.get("img"), state=obs["state"], noisy_action=x_t, t=t)

        # 2. 获取当前和下一步的 alpha_bar
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        
        # 边界处理：如果 t_next < 0，代表已经完成去噪过程，指向 alpha_bar=1 (无噪状态)
        if (t_next < 0).all():
            alpha_bar_t_next = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_t_next = self._extract(self.alphas_cumprod, t_next.clamp(min=0), x_t.shape)

        # 3. 预测原始动作 x0 (Action Clipping)
        # 这是为了确保模型每一步预测出的“干净动作”都在物理限速内，显著增强真机安全性
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0) 

        # 4. 根据 DDIM 公式计算前一步结果 x_prev
        # 当 eta=0 时，采样变为 ODE 求解，生成的轨迹最稳定
        sigma = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))
        dir_xt = torch.sqrt(1.0 - alpha_bar_t_next - sigma**2) * pred_noise

        x_prev = torch.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt
        
        # 如果设置了 eta > 0，则引入随机性
        if sigma > 0:
            x_prev += sigma * torch.randn_like(x_t)
            
        return x_prev