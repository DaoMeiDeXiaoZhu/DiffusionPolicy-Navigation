import os
import torch
from torch.utils.data import DataLoader

# 导入你自己写的模块
from dataset import ZarrDataset
from model import DiffusionTransformer
from diffusion import Diffusion
from trainer import Trainer


def main():
    # ============================================================
    # 1. 自动选择训练设备（GPU 优先）
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 2. 获取当前 train.py 所在目录
    #    这样路径不会因为你从哪里运行脚本而改变
    # ============================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ============================================================
    # 3. 拼接数据集路径
    #    数据存放在 DiffusionPolicy/collect_expert_experience/expert_data_collection
    # ============================================================
    data_dir = os.path.join(base_dir, "../collect_expert_experience/expert_data_collection")
    data_dir = os.path.normpath(data_dir)  # 规范化路径

    # ============================================================
    # 4. 创建 Dataset（从 zarr 读取数据）
    #    horizon=8 表示每个样本是 8 帧序列
    # ============================================================
    dataset = ZarrDataset(data_dir, horizon=8)

    # ============================================================
    # 5. 创建 DataLoader
    #    batch_size=2：每次训练 2 个序列
    #    shuffle=True：随机打乱
    #    num_workers=2：两个线程加载数据
    # ============================================================
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    # ============================================================
    # 6. 创建 Diffusion Policy 模型（Transformer）
    #    并移动到 GPU 或 CPU
    # ============================================================
    model = DiffusionTransformer().to(device)

    # ============================================================
    # 7. 创建 Diffusion 对象（包含 β、α、ᾱ、q_sample、p_sample）
    # ============================================================
    diff = Diffusion().to(device)

    # ============================================================
    # 8. 创建 Trainer（训练循环）
    # ============================================================
    trainer = Trainer(model, diff, loader, device=device)

    # ============================================================
    # 9. 开始训练（30 个 epoch）
    #    Trainer 内部会自动保存 latest_dp_model.pth
    # ============================================================
    trainer.train(epochs=30)


# ============================================================
# 10. Python 程序入口
# ============================================================
if __name__ == "__main__":
    main()
