import requests
import base64
import cv2
import numpy as np

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        _, buffer = cv2.imencode('.png', frame)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(
            "http://localhost:8111/detect",
            json={"image_data": image_data}
        )
        
        result = response.json()
        
        for detection in result:
            bbox = [int(coord) for coord in detection['bbox']]
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()

# Отредактировать эту строчку
process_video("data/crowd.mp4", "results/crowd_result.mp4")