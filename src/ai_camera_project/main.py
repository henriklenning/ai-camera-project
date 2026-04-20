import os
import shutil
import json
import cv2
from typing import Iterable, List, Dict
from ultralytics import YOLO
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
        return self.model(video_path, conf=conf, stream=True)
    
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

def main():
    path = input("Enter the path to the image or video: ")
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return

    # Generate a JSON filename based on the input path
    base_name = os.path.splitext(os.path.basename(path))[0]
    json_output = f"{base_name}_results.json"

    # Check if the file has already been analyzed
    if os.path.exists(json_output):
        print(f"File '{path}' has already been analyzed. Results are in '{json_output}'. skipping...")
        return

    is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        detector = VideoDetector()
        print(f"Analyzing video: {path}...")
        results = detector.detect(path)
        
        frame_count = detector.process_and_save(results, json_output)
        print(f"Processed {frame_count} frames. Results saved to {json_output}")
        
    else:
        detector = ImageDetector()
        print(f"Analyzing image: {path}...")
        results = detector.detect(path)
        
        detections = detector.process_results(results)
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"- {det['class_name']} ({det['confidence']:.2f})")
        
        # Save JSON results
        detector.save_json(detections, json_output)

if __name__ == "__main__":
    main()

