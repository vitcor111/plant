import os
from tkinter import Tk, filedialog
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import create_model
from utils import get_device
import matplotlib.pyplot as plt

# 获取设备
device = get_device()

# 创建模型
num_classes = 178  # 根据你的数据集类别数量替换
model = create_model(num_classes=num_classes)

# 加载模型权重
model_save_path = 'best_model.pth'
if not os.path.exists(model_save_path):
    raise FileNotFoundError("No pre-trained model found. Please train the model first.")
model.load_state_dict(torch.load(model_save_path))
model.to(device)
model.eval()  # 设置模型为评估模式


# 数据预处理
def get_transforms(mean=[0.4353, 0.3773, 0.2871], std=[0.2624, 0.2090, 0.2151]):
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),  # 使用自定义的均值和标准差
    ])


# 弹出文件选择对话框
root = Tk()
root.withdraw()  # 隐藏主窗口
image_path = filedialog.askopenfilename(title="Select an image",
                                        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))

if not image_path:
    print("No image selected. Exiting...")
else:
    # 加载和预处理图像
    transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度

    # 将图像移动到设备
    image_tensor = image_tensor.to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)

    # 获取类别名称（从数据集中提取）
    train_dir = 'data/train'  # 替换为你的训练集路径
    train_dataset = datasets.ImageFolder(root=train_dir)
    class_names = {v: k for k, v in train_dataset.class_to_idx.items()}

    # 显示结果
    predicted_class = class_names[predicted_label.item()]
    print(f"Predicted Class: {predicted_class}")

    # 可视化图像和预测结果
    plt.imshow(image)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')
    plt.show()



