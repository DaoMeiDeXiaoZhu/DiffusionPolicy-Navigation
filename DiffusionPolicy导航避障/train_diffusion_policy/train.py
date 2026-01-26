import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from dataset import ZarrDataset
from model import DiffusionTransformer
from diffusion import Diffusion
from trainer import Trainer
from config import config


def main():

    # 训练设备（从 config 加载）
    device = config["device"]

    # 创建 Dataset（从 zarr 读取数据，路径与 horizon 自动从 config 加载）
    dataset = ZarrDataset()

    # 创建 DataLoader（batchsize 从 config 加载）
    loader = DataLoader(
        dataset,
        batch_size=config["batchsize"],
        shuffle=True,
        num_workers=2
    )

    # 创建 Diffusion Policy 模型（Transformer）
    model = DiffusionTransformer().to(device)

    # 创建 Diffusion 对象（参数从 config 加载）
    diff = Diffusion().to(device)

    # 创建 Trainer（训练循环）
    trainer = Trainer(model, diff, loader)

    # 开始训练（epochs 从 config 加载）
    trainer.train(epochs=config["epochs"])


# Python 程序入口
if __name__ == "__main__":
    main()
