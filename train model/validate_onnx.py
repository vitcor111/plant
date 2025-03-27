import onnxruntime as ort
import numpy as np

# 显式指定使用 CPU 或 GPU 后端
ort_session = ort.InferenceSession(
    "plant-api/plant_model.onnx",
    providers=['CPUExecutionProvider']  # 或 ['CUDAExecutionProvider', 'CPUExecutionProvider']
)

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = ort_session.run(None, {"input": test_input})
print("输出维度:", outputs[0].shape)