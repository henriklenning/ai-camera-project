import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from picamera2_webstream import create_app
from ai_camera_project.stream_service import StreamService
import os
import shutil
import json
import cv2
from typing import Iterable, List, Dict
import sys

class BaseDetector:
    """
    Base class for object detection using YOLO models.
    """
    def __init__(self, model_path: str = "yolo26n_ncnn_model") -> None:
        self.model = YOLO(model_path, task='detect')

    def process_results(self, results: Iterable) -> List[Dict]:
        """
        Converts raw model output into a human-readable list of detections.
        """
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "box": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])]
                })
        return detections

    def annotate_frame(self, frame):
        """
        Processes a single frame, draws bounding boxes/labels, and returns the annotated image.
        """
        results = self.model(frame)
        r = results[0]
        annotated_frame = r.plot() 
        return annotated_frame
        
    def save_json(self, data: any, output_file: str):
        """
        Saves the given data as a JSON file.
        """
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to JSON: {output_file}")

class CameraDetector(BaseDetector):
    
    """
    Handles object detection for live camera streams.
    """
    def detect(self, source: int = 0, conf: float = 0.25) -> Iterable:
        return self.model(source, conf=conf, stream=True)

class VideoDetector(BaseDetector):
    """
    Handles object detection for video files.
    """
    def detect(self, video_path: str, conf: float = 0.25) -> Iterable:
        output = self.model(video_path, conf=conf, stream=True)
        return output
    
    def process_and_save(self, results: Iterable, output_file: str):
        """
        Processes video results and saves them to a JSON file.
        """
        all_detections = []
        for r in results:
            frame_detections = self.process_results([r])
            all_detections.append(frame_detections)
        
        self.save_json(all_detections, output_file)
        return len(all_detections)

class ImageDetector(BaseDetector):
    """
    Handles object detection for static images.
    """
    def detect(self, image_path: str, conf: float = 0.25) -> Iterable:
        return self.model(image_path, conf=conf)
    