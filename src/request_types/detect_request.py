from pydantic import BaseModel
from detector import detector
from PIL import Image
import base64
import io
import numpy as np

class DetectRequest(BaseModel):
    image_data: str = None

async def detect(detect_request: DetectRequest):
    image = np.array(Image.open(io.BytesIO(base64.b64decode(detect_request.image_data))))
    return detector.detect(image)

