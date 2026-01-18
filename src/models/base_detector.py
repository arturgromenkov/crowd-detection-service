from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def preprocess(self, image):
        pass
    
    @abstractmethod
    def inference(self, input_tensor):
        pass
    
    @abstractmethod
    def postprocess(self, predictions, 
                   orig_shape) -> list:
        pass
    
    def detect(self, image) -> list:
        input_tensor = self.preprocess(image)
        predictions = self.inference(input_tensor)
        return self.postprocess(predictions, image.shape)