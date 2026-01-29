import torch
import torch.nn as nn
from config import config

class DiffusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_steps = config["obs_steps"]
        self.horizon = config["horizon"]
        embed_dim = config["embed_dim"]
        
        # 1. 时间步编码 (FiLM Style)
        self.t_mlp = nn.Sequential(
            nn.Embedding(config["timesteps"], embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 2. 传感器编码器
        if config["use_lidar"]:
            self.state_fc = nn.Linear(config["lidar_dim"], embed_dim)
            # 位置编码：(1, obs_steps, embed_dim)
            self.obs_pos_emb = nn.Parameter(torch.zeros(1, self.obs_steps, embed_dim))

        if config["use_img"]:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 4, 2, 1), nn.Mish(),
                nn.Conv2d(16, 32, 4, 2, 1), nn.Mish(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.Mish(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            self.img_fc = nn.Linear(64, embed_dim)

        # 3. 动作编码器
        self.action_fc = nn.Linear(config["action_dim"], embed_dim)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, self.horizon, embed_dim))

        # 4. Transformer 核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=config["hidden_dim"],
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])

        # 5. 输出层
        self.out = nn.Linear(embed_dim, config["action_dim"])

    def forward(self, img, state, noisy_action, t):
        # 获取基础维度
        B = noisy_action.shape[0]
        T_act = noisy_action.shape[1] 

        # --- 1. 时间特征 (B, 1, D) ---
        t_feat = self.t_mlp(t).unsqueeze(1) # 结果必为 3 维

        # --- 2. 观测特征处理 ---
        obs_feat_list = []

        # 处理雷达 State
        if state is not None:
            # 【关键修复】如果 state 传入了 (B, T, 1, D) 或其他奇葩形状，强行转回 3 维
            if state.dim() != 3:
                # 假设最后一维是 lidar_dim
                state = state.view(B, -1, config["lidar_dim"])
            
            curr_T_obs = state.shape[1]
            s_feat = self.state_fc(state) # (B, curr_T_obs, D)
            # 加入位置编码 (截取当前序列长度)
            s_feat = s_feat + self.obs_pos_emb[:, :curr_T_obs, :]
            obs_feat_list.append(s_feat)

        # 处理图像 Img
        if config["use_img"] and img is not None:
            # 【关键修复】确保图像输入符合 (B, T, C, H, W) 5 维结构
            if img.dim() == 5:
                T_img = img.shape[1]
                img_reshaped = img.reshape(B * T_img, *img.shape[2:])
                i_feat = self.cnn(img_reshaped)
                i_feat = self.img_fc(i_feat).view(B, T_img, -1)
                obs_feat_list.append(i_feat)

        # 拼接观测特征并做一次“降维保护”
        # 确保 obs_feat 拼接后只能是 3 维
        obs_feat = torch.cat(obs_feat_list, dim=1)
        if obs_feat.dim() != 3:
            obs_feat = obs_feat.view(B, -1, config["embed_dim"])

        # --- 3. 动作特征 (B, T_act, D) ---
        act_feat = self.action_fc(noisy_action) + self.action_pos_emb[:, :T_act, :]

        # --- 4. 拼接所有 Tokens ---
        # 拼接顺序: [Time_Token, Obs_Tokens..., Action_Tokens...]
        # 维度检查：t_feat(3维), obs_feat(3维), act_feat(3维)
        tokens = torch.cat([t_feat, obs_feat, act_feat], dim=1)

        # --- 5. Transformer 运算 ---
        x = self.transformer(tokens)

        # --- 6. 提取预测噪声 ---
        # 我们只关心 Action Tokens 对应的输出部分
        # 索引计算：跳过 1(Time) + total_obs_tokens
        total_obs_tokens = obs_feat.shape[1]
        pred_noise = self.out(x[:, (1 + total_obs_tokens):, :])

        return pred_noise