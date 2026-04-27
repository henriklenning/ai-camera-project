from flask import Flask, Response
import logging
from time import sleep

app = Flask(__name__)
stream_instance = None

@app.route('/')
def index():
    return "AI Camera Stream is running. Go to /video_feed to see the stream."

@app.route('/video_feed')
def video_feed():
    return Response(
        _generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def _generate_frames():
    """Generator function to yield video frames"""
    global stream_instance
    if stream_instance is None:
        return
    
    while True:
        frame_data = None
        with stream_instance.lock:
            if stream_instance.frame_buffer is not None:
                frame_data = stream_instance.frame_buffer
        
        if frame_data is not None:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n'
                    b'\r\n' + frame_data + b'\r\n')
        
        # Use the framerate from the stream instance to pace the generator
        sleep(1/stream_instance.framerate)

def start(stream, port=8080, host='0.0.0.0'):
    global stream_instance
    stream_instance = stream
    logging.info(f"Starting web server on {host}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)

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
    
    start(stream)

