from flask import Flask, Response
import logging
from time import sleep

class WebServer:
    """
    Flask web server for streaming camera frames.
    """
    def __init__(self, stream_instance, host='0.0.0.0', port=8080):
        self.stream_instance = stream_instance
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        
        # Define routes inside __init__ so they have access to self and self.app
        @self.app.route('/video_feed')
        def video_feed():
            """Route to access the video stream"""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/')
        def index():
            """Route for the main page"""
            return """
            <html>
                <head>
                    <title>AI Camera Stream</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body { margin: 0; padding: 0; background: #1a1a1a; color: white; font-family: sans-serif; }
                        .container { 
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                        }
                        .stream-wrapper {
                            border: 4px solid #333;
                            border-radius: 8px;
                            overflow: hidden;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                        }
                        img { max-width: 100%; height: auto; display: block; }
                        h1 { margin-bottom: 20px; font-weight: 300; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>AI Live Stream</h1>
                        <div class="stream-wrapper">
                            <img src="/video_feed" alt="Camera Stream" />
                        </div>
                    </div>
                </body>
            </html>
            """

    def _generate_frames(self):
        """Generator function to yield video frames"""
        with self.stream_instance.clients_lock:
            self.stream_instance.clients += 1
            logging.info(f"Client connected. Total clients: {self.stream_instance.clients}")
        
        try:
            while True:
                frame_data = None
                with self.stream_instance.lock:
                    if self.stream_instance.frame_buffer is not None:
                        frame_data = self.stream_instance.frame_buffer
                
                if frame_data is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n'
                           b'\r\n' + frame_data + b'\r\n')
                
                # Use the framerate from the stream instance to pace the generator
                sleep(1/self.stream_instance.framerate)
                
        finally:
            with self.stream_instance.clients_lock:
                self.stream_instance.clients -= 1
                logging.info(f"Client disconnected. Remaining clients: {self.stream_instance.clients}")

    def run(self):
        """Starts the Flask server"""
        logging.info(f"Starting web server on {self.host}:{self.port}")
        # threaded=True allows multiple clients to view the stream
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

if __name__ == '__main__':
    # This block allows you to still test this file directly if needed
    from ai_camera_project.stream_service import StreamService
    
    stream = StreamService(
        width=1280,
        height=720,
        framerate=30,
        format="BGR888",
        brightness=0.0,
        contrast=0.03125,
        saturation=0.03125
    ).start()
    
    server = WebServer(stream)
    server.run()
