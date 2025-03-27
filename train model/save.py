# save_class_mapping.py
import os
import json
from pathlib import Path


def generate_class_mapping(train_dir, save_path="class_names.json"):
    """
    生成类别映射文件的独立脚本
    参数：
        train_dir: 训练集目录路径（例如 'data/train'）
        save_path: 映射文件保存路径（默认当前目录）
    """
    # 获取所有类别目录（按字母顺序排序）
    class_dirs = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"在 {train_dir} 中未找到任何类别目录")

    # 生成映射字典（与PyTorch的ImageFolder完全一致）
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(class_dirs)}

    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"成功生成映射文件：{save_path}")
    print("类别顺序验证：")
    for idx, name in idx_to_class.items():
        print(f"{idx}: {name}")


if __name__ == "__main__":
    # 使用示例（修改为你的实际路径）
    train_directory = "data/train"  # 替换为你的训练集路径
    generate_class_mapping(train_directory)