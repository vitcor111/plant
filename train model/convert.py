import torch
from model import create_model

# 加载训练好的模型
model = create_model(num_classes=178)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# 生成虚拟输入（必须与训练时一致）
dummy_input = torch.randn(1, 3, 224, 224)  # 输入尺寸：224x224 RGB

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "plant_model_opset13.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13  # ResNet需要较高版本
)