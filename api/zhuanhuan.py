import onnx
from onnx import version_converter

# 加载模型
model_path = "plant_model.onnx"
model = onnx.load(model_path)

# 检查原始模型的 Opset 版本
print(f"Original model opset version: {model.opset_import[0].version}")

# 转换模型的 Opset 版本到 13
try:
    converted_model = version_converter.convert_version(model, 13)
    print("Model successfully converted to Opset 13.")
except Exception as e:
    print(f"Failed to convert model: {e}")
    raise

# 保存转换后的模型
converted_model_path = "plant_model_opset13.onnx"
onnx.save(converted_model, converted_model_path)
print(f"Converted model saved to {converted_model_path}")