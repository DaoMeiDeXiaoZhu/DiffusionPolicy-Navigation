import sys
import os
import torch
import numpy as np
from collections import deque

# 确保能找到 config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class InferenceRunner:
    def __init__(self, model, diffusion):
        # 1. 基础配置对接
        self.device = torch.device(config["device"])
        self.model = model.eval().to(self.device)
        self.diff = diffusion
        
        # 2. 从 GLOBAL_SETTINGS 同步维度
        self.horizon = config["horizon"]
        self.obs_steps = config["obs_steps"] # 【关键：对齐观察步数】
        self.action_dim = config["action_dim"]
        self.n_steps = config.get("inference_steps", 50)

        # 3. 【新增：滑动窗口缓存】
        self.obs_history = deque(maxlen=self.obs_steps)

    def _prepare_obs(self, obs_raw):
        # 1. 强制将输入的雷达压平为 1 维 (360,)，不管外面传进来是什么形状
        state = obs_raw['state']
        if isinstance(state, np.ndarray):
            state = state.flatten() 
            
        # 归一化处理
        state = np.clip(state, config['lidar_min'], config['lidar_max'])
        state = (state - config['lidar_min']) / (config['lidar_max'] - config['lidar_min'])
        
        # 2. 存入滑动窗口
        self.obs_history.append(state)
        while len(self.obs_history) < self.obs_steps:
            self.obs_history.appendleft(state)
            
        # 3. 构造 Tensor：(4, 360) -> unsqueeze -> (1, 4, 360) 
        # 这样就严格对应训练时的 (Batch, Time, Dim)
        obs_np = np.array(list(self.obs_history), dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_np).to(self.device)
        
        obs_dict = {
            "state": obs_tensor.unsqueeze(0), # 增加 Batch 维，变为 3 维
            "img": None # 必须显式为 None
        }
        return obs_dict
    
    def _unnormalize_action(self, action_norm):
        """反归一化：[-1, 1] -> [min, max]"""
        v_norm, w_norm = action_norm[0], action_norm[1]
        
        stats = config['action_stats']
        v = (v_norm + 1) / 2 * (stats['v_max'] - stats['v_min']) + stats['v_min']
        w = (w_norm + 1) / 2 * (stats['w_max'] - stats['w_min']) + stats['w_min']
        
        return np.array([v, w], dtype=np.float32)

    @torch.no_grad()
    def predict_action(self, obs_raw):
        # 1. 准备数据 (包含历史帧)
        obs = self._prepare_obs(obs_raw)
        
        # 2. 生成 DDIM 时间序列
        times = torch.linspace(config["timesteps"] - 1, 0, self.n_steps + 1, device=self.device).long()
        
        # 3. 初始化随机噪声 (1, horizon, 2)
        x = torch.randn(1, self.horizon, self.action_dim, device=self.device)

        # 4. DDIM 确定性迭代去噪
        for i in range(self.n_steps):
            t = times[i].unsqueeze(0)
            t_next = times[i+1].unsqueeze(0)
            # 调用已修正的 diffusion 类
            x = self.diff.ddim_sample(self.model, obs, x, t, t_next)

        # 5. 后处理
        # 策略：取预测序列的第一步执行 (Action Chunking 思想)
        action_norm = x[0, 0].cpu().numpy()
        return self._unnormalize_action(action_norm)