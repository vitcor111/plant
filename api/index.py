import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import traceback
import os
from scipy.special import softmax

# 打印当前工作目录和文件列表
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir()}")

# 全局配置
try:
    with open('class_names.json') as f:
        CLASS_NAMES = json.load(f)
    with open('config.json') as f:
        config = json.load(f)
except Exception as e:
    print(f"配置文件加载失败: {e}")
    raise

# ONNX会话单例
ort_session = None


def get_ort_session():
    global ort_session
    if ort_session is None:
        try:
            model_path = 'plant_model_opset13.onnx'
            ort_session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    return ort_session


def preprocess(image_data):
    """图像预处理"""
    try:
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = img.resize((config['img_size'], config['img_size']))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - config['mean']) / config['std']
        return img_array.transpose(2, 0, 1)[np.newaxis, ...]
    except Exception as e:
        print(f"预处理失败: {e}")
        raise


def main_handler(event, context):
    try:
        # 获取二进制图片数据（无需Base64解码）
        if isinstance(event, dict) and 'body' in event:
            # 如果通过API网关触发且body是Base64字符串
            image_data = base64.b64decode(event['body'])
        else:
            # 直接接收二进制数据
            image_data = event if isinstance(event, bytes) else event.encode()

        # 预处理和推理
        input_tensor = preprocess(image_data).astype(np.float32)
        session = get_ort_session()
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        probabilities = softmax(outputs[0][0])
        pred_idx = np.argmax(probabilities)

        # 构建响应
        return {
            "isBase64Encoded": False,
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "class": CLASS_NAMES[str(pred_idx)],
                "confidence": round(float(probabilities[pred_idx]), 4)
            })
        }
    except Exception as e:
        print(f"处理失败: {str(e)}\n{traceback.format_exc()}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "内部错误"})
        }


# 本地测试
if __name__ == "__main__":
    with open("test_rose.jpg", "rb") as f:
        test_image = f.read()
    print(main_handler(test_image, None))