import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import Adam
from tqdm import tqdm
import os
from config import config


class Trainer:
    def __init__(self, model, diffusion, dataloader):
        # 从 config 加载训练设备
        self.device = config["device"]

        # 噪声预测网络 εθ
        self.model = model.to(self.device)

        # Diffusion 对象（包含 q_sample / p_sample）
        self.diff = diffusion

        # DataLoader（批量提供 obs + action）
        self.loader = dataloader

        # Adam 优化器（学习率从 config 加载）
        self.opt = Adam(model.parameters(), lr=config["learning_rate"])

        # AMP 混合精度缩放器（节省显存并加速）
        self.scaler = torch.cuda.amp.GradScaler()

        # 模型保存路径（从 config 加载）
        self.save_path = config["model_path"]

        # 模态选择开关（从 config 加载）
        self.use_img = config["use_img"]
        self.use_lidar = config["use_lidar"]

        # 梯度裁剪阈值
        self.max_norm = config["max_norm"]

    def train(self, epochs):

        for ep in range(epochs):
            
            pbar = tqdm(self.loader)
            running_loss = 0.0   # ★ 新增：记录累计损失

            for i, (obs, action) in enumerate(pbar):

                obs = {k: v.to(self.device) for k, v in obs.items()}
                action = action.to(self.device)

                B, T, _ = action.shape
                t = torch.randint(0, self.diff.T, (B,), device=self.device)

                with torch.cuda.amp.autocast():
                    noisy_action, noise = self.diff.q_sample(action, t)

                    pred = self.model(
                        img=obs["img"],
                        state=obs["state"],
                        noisy_action=noisy_action,
                        t=t,
                        use_img=self.use_img,
                        use_lidar=self.use_lidar
                    )

                    loss = torch.mean((pred - noise)**2)

                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                # ★ 更新平均损失
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)

                # ★ tqdm 显示平均损失
                pbar.set_description(f"Epoch {ep} AvgLoss {avg_loss:.4f}")

            torch.save(self.model.state_dict(), self.save_path)
            print(f"[保存] 模型已保存到 {self.save_path}")

