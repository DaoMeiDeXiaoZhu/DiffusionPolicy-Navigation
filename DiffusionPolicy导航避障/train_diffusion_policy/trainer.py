import torch
from torch.optim import Adam
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, diffusion, dataloader, lr=3e-5, device="cuda"):
        # 训练用的噪声预测网络 εθ
        self.model = model
        self.model.to(device)

        # Diffusion 对象（包含 q_sample / p_sample）
        self.diff = diffusion

        # DataLoader（批量提供 obs + action）
        self.loader = dataloader

        # Adam 优化器
        self.opt = Adam(model.parameters(), lr=lr)

        # 训练设备（GPU / CPU）
        self.device = device

        # AMP 混合精度缩放器（节省显存 + 加速）
        self.scaler = torch.cuda.amp.GradScaler()

        # 模型保存路径（与 train.py 同目录）
        self.save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "latest_dp_model.pth"
        )

    def train(self, epochs=20):
        # ============================
        # 主训练循环（共 epochs 轮）
        # ============================
        for ep in range(epochs):

            # tqdm 进度条显示
            pbar = tqdm(self.loader)

            # ============================
            # 遍历每个 batch
            # ============================
            for obs, action in pbar:

                # 将 obs（img + state）移动到 GPU
                obs = {k: v.to(self.device) for k, v in obs.items()}

                # 将专家动作移动到 GPU
                action = action.to(self.device)

                # B = batch size, T = horizon, _ = action_dim
                B, T, _ = action.shape

                # 随机采样扩散时间步 t ∈ [0, T-1]
                # shape = (B,)
                t = torch.randint(0, self.diff.T, (B,), device=self.device)

                # ============================
                # 前向传播（使用混合精度）
                # ============================
                with torch.cuda.amp.autocast():

                    # ---------------------------------------------------
                    # 1. 前向扩散：q(x_t | x_0)
                    #    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
                    #    noisy_action = x_t
                    #    noise = ε（训练时的 ground truth）
                    # ---------------------------------------------------
                    noisy_action, noise = self.diff.q_sample(action, t)

                    # ---------------------------------------------------
                    # 2. 模型预测噪声 εθ(x_t, o, t)
                    #    pred = εθ
                    # ---------------------------------------------------
                    pred = self.model(
                        img=obs["img"],
                        state=obs["state"],
                        noisy_action=noisy_action,
                        t=t
                    )

                    # ---------------------------------------------------
                    # 3. 训练目标：MSE(εθ, ε)
                    #    对应 Diffusion Policy 的核心损失
                    # ---------------------------------------------------
                    loss = torch.mean((pred - noise)**2)

                # ============================
                # 反向传播 + 优化
                # ============================
                self.opt.zero_grad()

                # 使用 AMP 缩放 loss
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 更新参数
                self.scaler.step(self.opt)

                # 更新缩放器
                self.scaler.update()

                # 更新进度条显示
                pbar.set_description(f"Epoch {ep} Loss {loss.item():.4f}")

            # ============================
            # 每个 epoch 结束后保存一次模型
            # 覆盖之前的 latest_dp_model.pth
            # ============================
            torch.save(self.model.state_dict(), self.save_path)
            print(f"[保存] 模型已保存到 {self.save_path}")
