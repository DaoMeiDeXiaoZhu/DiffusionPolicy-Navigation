import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR # 新增：调度器
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class Trainer:
    def __init__(self, model, diffusion, dataloader):
        self.device = torch.device(config["device"])
        self.save_path = config["model_save_path"]
        
        self.model = model.to(self.device)
        if config.get("gpu_count", 0) > 1:
            print(f"检测到 {config['gpu_count']} 个 GPU，开启 DataParallel")
            self.model = nn.DataParallel(self.model)

        if os.path.exists(self.save_path):
            print(f"[加载] 正在恢复训练: {self.save_path}")
            state_dict = torch.load(self.save_path, map_location=self.device)
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
        
        self.diff = diffusion
        self.loader = dataloader
        
        self.opt = AdamW(self.model.parameters(), lr=config["learning_rate"], weight_decay=1e-6)
        
        # 新增：余弦退火调度器，帮助后期收敛
        self.scheduler = CosineAnnealingLR(self.opt, T_max=config["epochs"])
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        self.max_norm = config["max_norm"]

    def train(self):
        epochs = config["epochs"]
        
        for ep in range(epochs):
            self.model.train() 
            pbar = tqdm(self.loader, desc=f"Epoch {ep+1}/{epochs}")
            running_loss = 0.0

            for i, (obs, action) in enumerate(pbar):
                obs = {k: v.to(self.device) if v is not None else None for k, v in obs.items()}
                action = action.to(self.device)

                B = action.shape[0]
                t = torch.randint(0, self.diff.T, (B,), device=self.device).long()

                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    # 前向加噪
                    noisy_action, noise = self.diff.q_sample(action, t)

                    # 预测噪声：此时模型已适配 (obs_steps -> horizon) 架构
                    pred = self.model(
                        img=obs.get("img"),
                        state=obs.get("state"),
                        noisy_action=noisy_action,
                        t=t
                    )

                    loss = nn.functional.mse_loss(pred, noise)

                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                
                self.scaler.unscale_(self.opt)
                # 梯度裁剪：Diffusion 模型训练必备，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                
                self.scaler.step(self.opt)
                self.scaler.update()

                running_loss += loss.item()
                # 实时显示学习率和 Loss
                pbar.set_postfix(
                    loss=f"{running_loss/(i+1):.6f}", 
                    lr=f"{self.opt.param_groups[0]['lr']:.2e}"
                )

            # 更新学习率
            self.scheduler.step()

            # 保存模型
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(raw_model.state_dict(), self.save_path)