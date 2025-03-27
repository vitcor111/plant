import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch


def compute_mean_std(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


if __name__ == "__main__":
    train_dir = 'data/train'  # 替换为你的训练集路径
    mean, std = compute_mean_std(train_dir)
    print(f"Mean: {mean}")
    print(f"Std: {std}")



