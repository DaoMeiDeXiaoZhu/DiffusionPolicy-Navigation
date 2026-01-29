import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 路径对齐
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ZarrDataset
from model import DiffusionTransformer
from diffusion import Diffusion
from trainer import Trainer
from config import config

def main():
    # 1. 环境初始化
    device = torch.device(config["device"])
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 同步 config 里的 gpu_count，防止配置冲突
    print(f'>>> 运行环境: {device} | 可用GPU: {gpu_count} | 目标Epochs: {config["epochs"]}')
    print(f'>>> 序列长度(Horizon): {config["horizon"]} | 批大小: {config["batchsize"]}')

    # 2. 数据准备
    dataset = ZarrDataset()
    
    # 算力分配：num_workers 通常设为核心数的 1/2 或 4*gpu_count
    num_workers = min(os.cpu_count(), 4 * max(1, gpu_count))
    
    loader = DataLoader(
        dataset,
        batch_size=config["batchsize"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if gpu_count > 0 else False,
        drop_last=True  # 建议开启：防止最后一个不满 batch 的碎片干扰梯度
    )

    # 3. 模型与扩散器初始化
    model = DiffusionTransformer().to(device)
    diff = Diffusion().to(device) # 暂时保持单体，由 Trainer 内部处理

    # 4. 多卡并行逻辑 (针对模型主体)
    if gpu_count > 1:
        print(f"[并行] 检测到多张显卡，开启 nn.DataParallel 模式")
        model = nn.DataParallel(model)
        # 注意：diff 对象建议不要包装，除非它内部包含需要学习的参数 (Parameters)
        # 如果它只包含 Buffer (alpha_bar)，DataParallel 访问起来会很麻烦

    # 5. 启动训练器
    print(">>> 训练引擎启动...")
    trainer = Trainer(model, diff, loader)
    trainer.train() 


if __name__ == "__main__":
    main()