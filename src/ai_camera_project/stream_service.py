import threading
from picamera2 import Picamera2
import logging
import io
import cv2
import numpy as np
from time import sleep, time

from libcamera import Transform

class StreamService:
    def __init__(self, width=1280, height=720, framerate=30, format="RGB888",
                 brightness=0.0, contrast=1.0, saturation=1.0, detector=None):
        self.resolution = (width, height)
        self.lock = threading.Lock()
        self.frame_buffer = None
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.framerate = framerate
        self.detector = detector
        
        # Get the camera index using the utility function
        camera_index = 0
        logging.info(f"Using camera at index {camera_index}")
        self.picam2 = Picamera2(camera_index)
        
        try:
            # Use RGB888 as it's more standard for capture_array
            # We will convert to BGR for OpenCV processing
            config = self.picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                buffer_count=4,
                transform=Transform(vflip=1, hflip=1)
            )
            
            # Configure the camera
            self.picam2.configure(config)
            
            # Apply camera controls after configuration
            self.set_camera_properties(brightness, contrast, saturation)
            
            self.buffer = io.BytesIO()
            logging.info("Camera configuration complete")
        except Exception as e:
            logging.error(f"Error configuring camera: {e}")
            raise
            
    def set_camera_properties(self, brightness, contrast, saturation):
        """
        Set camera properties safely using libcamera controls
        """
        try:
            controls = {}

            # Explicitly set AWB to Auto (0 usually means Auto in libcamera)
            controls["AwbEnable"] = True
            controls["AwbMode"] = 0 

            # Brightness: Range is typically -1.0 to 1.0, default 0.0
            try:
                logging.info(f"Setting brightness to {brightness}")
                controls["Brightness"] = float(brightness)
            except Exception as e:
                logging.warning(f"Could not set brightness: {e}")
                
            # Contrast: Range is typically 0.0 to 32.0, default 1.0
            try:
                logging.info(f"Setting contrast to {contrast}")
                controls["Contrast"] = float(contrast)
            except Exception as e:
                logging.warning(f"Could not set contrast: {e}")
                
            # Saturation: Range is typically 0.0 to 32.0, default 1.0
            try:
                logging.info(f"Setting saturation to {saturation}")
                controls["Saturation"] = float(saturation)
            except Exception as e:
                logging.warning(f"Could not set saturation: {e}")
                
            # Apply controls if any were set
            if controls:
                logging.info(f"Applying camera controls: {controls}")
                self.picam2.set_controls(controls)
                
        except Exception as e:
            logging.warning(f"Error setting camera properties: {e}")
        
    def start(self):
        """Start the video streaming thread"""
        try:
            self.picam2.start()
            # Wait for AWB and AGC to settle
            logging.info("Waiting for camera to settle (AWB/AGC)...")
            sleep(2.0)
            
            success = self._capture_single_frame()
            
            if not success:
                logging.error("Failed to capture initial frame")
                return None
                
            self.capture_thread = threading.Thread(target=self._capture_frames, 
                                               daemon=True, 
                                               name="CaptureThread")
            self.capture_thread.start()
            logging.info("Video stream started successfully")
            return self
        except Exception as e:
            logging.error(f"Error starting camera: {e}")
            return None
        
    def _capture_single_frame(self):
        """Capture a single frame"""
        try:
            # For the initial frame, we can just use capture_file (JPEG)
            self.buffer.seek(0)
            self.buffer.truncate()
            self.picam2.capture_file(self.buffer, format='jpeg')
            self.frame_buffer = self.buffer.getvalue()
            return True
        except Exception as e:
            logging.error(f"Error capturing initial frame: {str(e)}")
            return False
        
    def stop(self):
        """Stop the video streaming"""
        self.stop_event.set()
        if hasattr(self, 'picam2'):
            try:
                self.picam2.stop()
            except Exception as e:
                logging.error(f"Error stopping camera: {e}")
        
    def _capture_frames(self):
        """Continuously capture frames from the camera"""
        frame_interval = 1/self.framerate
        retries = 0
        max_retries = 3

        while not self.stop_event.is_set():
            try:
                start_time = time()

                if self.detector:
                    # Capture as array (RGB888 configuration gives BGR array)
                    frame = self.picam2.capture_array()
                    
                    # Process and annotate
                    annotated_frame = self.detector.annotate_frame(frame)
                    
                    # Encode back to JPEG for the web stream
                    _, jpeg_encoded = cv2.imencode('.jpg', annotated_frame)
                    jpeg_data = jpeg_encoded.tobytes()
                else:
                    self.buffer.seek(0)
                    self.buffer.truncate()
                    self.picam2.capture_file(self.buffer, format='jpeg')
                    jpeg_data = self.buffer.getvalue()

                with self.lock:
                    self.frame_buffer = jpeg_data

                if self.frame_count % 300 == 0:
                    logging.info(f"Stream stats - Frame: {self.frame_count}, "
                                    f"Size: {len(jpeg_data)} bytes, ")
                self.frame_count += 1
                retries = 0  # Reset retries on success

                # Maintain frame rate
                elapsed = time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    sleep(sleep_time)

            except RuntimeError as e:
                logging.error(f"Runtime error during capture: {e}")
                retries += 1
                if retries >= max_retries:
                    logging.error("Max retries exceeded. Restarting camera...")
                    try:
                        self.picam2.stop()
                        sleep(1)  # Wait before restarting
                        self.picam2.start()
                        retries = 0
                    except Exception as restart_error:
                        logging.error(f"Error restarting camera: {restart_error}")
                        sleep(5)  # Give more time before next attempt
            except Exception as e:
                logging.error(f"Unexpected error during capture: {e}")
                sleep(0.1)