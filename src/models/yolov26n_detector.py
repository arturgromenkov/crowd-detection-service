import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
from models.base_detector import BaseDetector
from settings import YOLOV26N_CONFIDENCE

class YOLOV26NPersonDetector(BaseDetector):
    def __init__(self):
        self.model_path = "src/models/weights/yolov26n.onnx"
        self.conf_threshold = YOLOV26N_CONFIDENCE
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = 640
        
        self.load_model()
    
    def load_model(self):
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = input_shape[2]
    
    def preprocess(self, image) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch
    
    def inference(self, input_tensor) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]
    
    def postprocess(self, predictions, orig_shape) -> List[dict]:
        results = []
        pred = predictions[0]  # Remove batch dimension
        
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        # Assuming output format: [x1, y1, x2, y2, confidence, class_id]
        for det in pred:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, class_id = det[:6]
                
                if conf > self.conf_threshold and int(class_id) == 0: # Person class
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_name': 'person'
                    })
        
        return results
    