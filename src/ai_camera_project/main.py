import os
import json
import cv2
from typing import Iterable, List, Dict
import sys

# Custom modules
from ai_camera_project.detector import BaseDetector, CameraDetector, ImageDetector, VideoDetector
from ai_camera_project.stream_service import StreamService
from ai_camera_project.web_server import WebServer


def main():
    user_input = input("Enter the path to the image/video or type 'camera' for live stream:").strip()
    path_lower = user_input.lower()

    if path_lower == 'camera':
        print("Starting live camera feed with detection...")
        detector = CameraDetector()
        
        stream = StreamService(
            width=1280,
            height=720,
            framerate=30,
            format="BGR888",
            brightness=0.0,
            contrast=0.03125,
            saturation=0.03125,
            detector=detector
        ).start()
    
        if stream:
            server = WebServer(stream)
            server.run()
        else:
            print("Error: Could not start camera stream")
        return
    
    if not os.path.exists(user_input):
        print(f"Error: File '{user_input}' not found")
        return
    

    # Generate a JSON filename based on the input path
    base_name = os.path.splitext(os.path.basename(user_input))[0]
    json_output = f"{base_name}_results.json"

    # Check if the file has already been analyzed
    if os.path.exists(json_output):
        print(f"File '{user_input}' has already been analyzed. Results are in '{json_output}'. skipping...")
        return

    is_video = user_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        detector = VideoDetector()
        print(f"Analyzing video: {user_input}...")
        results = detector.detect(user_input)
        
        frame_count = detector.process_and_save(results, json_output)
        print(f"Processed {frame_count} frames. Results saved to {json_output}")
        
    else:
        detector = ImageDetector()
        print(f"Analyzing image: {user_input}...")
        results = detector.detect(user_input)
        
        detections = detector.process_results(results)
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"- {det['class_name']} ({det['confidence']:.2f})")
        
        # Save JSON results
        detector.save_json(detections, json_output)

if __name__ == "__main__":
    main()

