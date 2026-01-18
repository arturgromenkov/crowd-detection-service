from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Или yolo11s.pt, yolo11m.pt и т.д.

model.export(
    format='onnx',
    imgsz=640,
    nms=True,  # Встроить NMS в модель
    opset=17,
    simplify=True
)

print("Экспорт завершен. Файл: yolov11n.onnx")