import os
from dataset import load_datasets, create_data_loaders, show_sample_images
from model import create_model
from train import train_model, evaluate_model
from utils import get_device
import torch.nn as nn
import torch.optim as optim
import torch  # 确保导入 torch 模块

# 设置数据集路径
train_dir = 'data/train'  # 替换为你的训练集路径
valid_dir = 'data/valid'  # 替换为你的验证集路径

# 加载和预处理数据
train_dataset, val_dataset = load_datasets(train_dir, valid_dir)
train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size=32)

# 获取类别名称（从数据集中提取）
class_names = {v: k for k, v in train_dataset.class_to_idx.items()}

# 显示样本图像
show_sample_images(train_loader, class_names)

# 获取设备
device = get_device()

# 创建模型
model = create_model(num_classes=len(class_names))
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和评估参数
num_epochs = 20

# 模型保存路径
model_save_path = 'best_model.pth'

# 初始化最佳验证准确率
best_val_acc = 0.0

# 检查是否已经存在预训练模型
if os.path.exists(model_save_path):
    print(f"Loading pre-trained model from {model_save_path}...")
    model.load_state_dict(torch.load(model_save_path))
    print("Pre-trained model loaded successfully.")
else:
    print("No pre-trained model found. Starting training from scratch.")

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    # 训练模型
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 评估模型
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 更新最佳验证准确率
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"Best validation accuracy improved to {best_val_acc:.4f}. Model saved to {model_save_path}.")

print("训练完成！")



