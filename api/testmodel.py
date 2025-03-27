import onnx
model = onnx.load("plant_model.onnx")
onnx.checker.check_model(model)
print("Model is valid!")