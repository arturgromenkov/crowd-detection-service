from dotenv import load_dotenv
import os

load_dotenv()  # по умолчанию ищет .env в текущей директории

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8022"))

MODEL = os.getenv("MODEL", "yolov26n")

YOLOV26N_CONFIDENCE = float(os.getenv("YOLOV26N_CONFIDENCE", 0))
YOLOV11N_CONFIDENCE = float(os.getenv("YOLOV11N_CONFIDENCE", 0))
