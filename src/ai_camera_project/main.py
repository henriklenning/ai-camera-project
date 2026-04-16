from typing import Iterable

from ultralytics import YOLO
import sys

class CameraDetector:
    """
    Handles object detection for live camera streams (e.g., Raspberry Pi Camera).
    """
    def __init__(self, model_path:str = "yolo26n_ncnn_model") -> None:
        """
        Initialize the camera detector.
        :param model_path: Path to the exported NCNN model directory.
        """
        self.model = YOLO(model_path)

    def detect(self, source:int = 0, conf:float = 0.25) -> Iterable:
        """
        Run detection on a live video source.
        :param source: Camera index (0 is usually the default camera) or video file path.
        :param conf: Confidence threshold (0.0 to 1.0).
        :return: Raw YOLO Results object.
        """
        # The model automatically handles the loop for video sources
        return self.model(source, conf=conf)

class ImageDetector():
    """
    Handles object detection for static images.
    """
    def __init__(self, model_path:str = "yolo26n_ncnn_model") -> None:
        """
        Initialize the image detector.
        :param model_path: Path to the exported NCNN model directory.
        """
        self.model = YOLO(model_path)
        
    def detect(self, image_path:str, conf:float = 0.25) -> Iterable:
        """
        Analyze a single image.
        :param image_path: Path to the image file (e.g., 'bus.jpg').
        :param conf: Confidence threshold (0.0 to 1.0).
        :return: Raw YOLO Results object.
        """
        results = self.model(image_path, conf)
        
        return results

    def process_results(self, results:Iterable) -> list[dict]:
        """
        Converts raw model output into a human-readable list of detections.
        :param results: The Results object returned by the detect() method.
        :return: A list of dictionaries containing box coordinates, confidence, and class names.
        """
        detections = []
        for r in results:
            # Iterate through each bounding box found in the image
            for box in r.boxes:
                detections.append({
                    "box": box.xyxy[0].tolist(),  # Coordinates: [xmin, ymin, xmax, ymax]
                    "confidence": float(box.conf[0]), # Probability score
                    "class_id": int(box.cls[0]), # Numerical ID of the object category
                    "class_name": r.names[int(box.cls[0])] # Human-readable label (e.g., 'person')
                })
        return detections

    def save_results(self, results:Iterable, output_path:str = "output.jpg") -> str:
        """
        Saves a copy of the image with bounding boxes and labels drawn on it.
        :param results: The Results object returned by the detect() method.
        :param output_path: Where to save the annotated image.
        :return: The path where the image was saved.
        """
        for r in results:
            r.save(filename=output_path)
        return output_path

if __name__ == "__main__":
    # This block only runs if the script is executed directly (e.g., python main.py bus.jpg)
    if len(sys.argv) > 1:
        img_path = sys.argv[1] # Take the image path from the command line argument
        detector = ImageDetector()
        
        print(f"Analyzing {img_path}...")
        results:Iterable = detector.detect(img_path)
        
        # Extract and print readable detection info
        detections = detector.process_results(results)
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"- {det['class_name']} ({det['confidence']:.2f})")
        
        # Save the visualization to disk
        output = detector.save_results(results)
        print(f"Results saved to {output}")
    else:
        print("Usage: python src/ai_camera_project/main.py <path_to_image>")
