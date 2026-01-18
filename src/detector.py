from settings import MODEL
from models.yolov26n_detector import YOLOV26NPersonDetector

if MODEL == 'yolov26n':
    detector = YOLOV26NPersonDetector()
