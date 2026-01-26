import torch

class InferenceRunner:
    def __init__(self, model, diffusion, device="cuda"):
        # 将模型移动到推理设备（CPU 或 GPU）
        self.model = model.to(device)

        # Diffusion 对象（包含 betas、alphas、alpha_bar 以及 p_sample）
        self.diff = diffusion

        # 推理设备
        self.device = device

    @torch.no_grad()  # 推理阶段不需要梯度，节省显存和加速
    def predict_action(self, obs):
        # ---------------------------------------------------------
        # 1. 定义动作序列长度 T（与训练时 horizon 一致）
        #    这里固定为 16，对应 x_t 的时间维度
        # ---------------------------------------------------------
        T = 16

        # ---------------------------------------------------------
        # 2. 初始化 x_T 为标准正态分布噪声
        #    shape = (batch=1, T, action_dim=2)
        #    这是反向扩散的起点
        # ---------------------------------------------------------
        x = torch.randn(1, T, 2, device=self.device)

        # ---------------------------------------------------------
        # 3. 从 t = T-1 到 0 逐步去噪
        #    对应反向扩散公式：
        #    x_{t-1} = p_sample(model, x_t, t)
        # ---------------------------------------------------------
        for t in reversed(range(self.diff.T)):
            # 当前扩散步 t（shape = (1,)）
            tt = torch.tensor([t], device=self.device)

            # 执行一步反向扩散：x_t → x_{t-1}
            x = self.diff.p_sample(
                self.model,  # 噪声预测网络 εθ
                obs,         # 观测（图像 + 雷达）
                x,           # 当前带噪动作 x_t
                tt           # 当前时间步 t
            )

        # ---------------------------------------------------------
        # 4. 返回最终动作序列 x_0 的第 0 帧动作
        #    shape = (2,) → numpy
        #    这是机器人当前时刻要执行的动作 (v, w)
        # ---------------------------------------------------------
        return x[0, 0].cpu().numpy()
