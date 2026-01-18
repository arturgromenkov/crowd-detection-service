import requests
import base64
import io
import os
from datetime import datetime
from PIL import Image, ImageDraw

with open("data/image.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8111/detect",
    json={"image_data": image_data}
)

result = response.json()
print(result)

image = Image.open("data/image.png")
draw = ImageDraw.Draw(image)

for detection in result:
    bbox = detection['bbox']
    confidence = detection['confidence']
    class_name = detection['class_name']
    
    draw.rectangle(bbox, outline="red", width=2)
    draw.text((bbox[0], bbox[1] - 10), f"{class_name} {confidence:.2f}", fill="red")

image.save("results/result.png")
print("Saved to results/result.png")