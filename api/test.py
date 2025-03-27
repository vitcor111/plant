import requests

url = "https://1347661153-6fj0js34pe.ap-shanghai.tencentscf.com"
headers = {
    "Content-Type": "image/jpeg",
    "Authorization": "Bearer YOUR_TOKEN"  # 如果启用鉴权
}

with open("test_rose.jpg", "rb") as f:
    response = requests.post(url, data=f.read(), headers=headers)

print(response.status_code)
print(response.json())