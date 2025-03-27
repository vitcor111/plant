import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

# 数据预处理
def get_transforms(mean=[0.4353, 0.3773, 0.2871], std=[0.2624, 0.2090, 0.2151]):
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),  # 使用自定义的均值和标准差
    ])

# 加载训练集和验证集
def load_datasets(train_dir, valid_dir):
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    return train_dataset, val_dataset

# 创建DataLoader
def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# 显示样本图像
def show_sample_images(loader, class_names):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        img = images[i].permute(1, 2, 0)  # 将CWH转为HWC以便显示
        mean = torch.tensor([0.4353, 0.3773, 0.2871])
        std = torch.tensor([0.2624, 0.2090, 0.2151])
        img = img * std + mean  # 反归一化
        ax.imshow(img.numpy())
        ax.set_title(f"Class: {class_names[labels[i].item()]}")
        ax.axis('off')
    plt.show()



