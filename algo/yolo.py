# -*- coding: utf-8 -*-
# @Time    : 2025/7/9 10:13
# @Author  : Chaohe Wen
# @Email   : wenchaohe@tsingtec.com

from ultralytics import YOLO


class YoloPoseEstimation:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None
    
    def infer(self, input):
        self.result = self.model.track(input, tracker="bytetrack.yaml", conf=0.4, persist=True, verbose=False)
        return self.result[0]
    
    def info(self):
        # Process results list
        for res in self.result:
            boxes = res.boxes  # Boxes object for bbox outputs
            masks = res.masks  # Masks object for segmentation masks outputs
            keypoints = res.keypoints  # Keypoints object for pose outputs
            probs = res.probs  # Probs object for classification outputs

            print(f"Boxes : {boxes}")
            print(f"Masks : {masks}")
            print(f"Keypoints : {keypoints}")
            print(f"Probs : {probs}")

class CigaretteDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None
    
    def infer(self, input):
        self.result = self.model.track(input, conf=0.5, persist=True, verbose=False)
        if self.result[0].boxes is None:
            return str(False)
        else:
            return str(True)

class FaceDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None
    
    def infer(self, input):
        self.result = self.model(input, conf=0.5, verbose=False)
        return self.result[0].boxes.xyxy.cpu().squeeze().tolist()
        
class PackageDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None
    
    def infer(self, input):
        self.result = self.model(input, conf=0.5, verbose=False)
        return self.result[0].boxes.xyxy.cpu().squeeze().tolist()