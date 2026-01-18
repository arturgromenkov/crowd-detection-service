from dotenv import load_dotenv
import os

load_dotenv()  # по умолчанию ищет .env в текущей директории

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8022"))

MODEL = os.getenv("MODEL")
