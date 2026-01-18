from settings import MODEL
from models.yolov26n_detector import YOLOV26NPersonDetector
from models.yolov11n_detector import YOLOV11NPersonDetector

if MODEL == 'yolov26n':
    detector = YOLOV26NPersonDetector()
elif MODEL == 'yolov11n':
    detector = YOLOV11NPersonDetector()
