from picamera2_webstream import create_app
from ai_camera_project.stream_service import StreamService

stream = StreamService(
    width=1280,
    height=720,
    framerate=30,
    format="RGB888",
    # 1.0 / 32.0 = 0.03125 (Identity for libcamera on Pi 5)
    contrast=0.03125,
    saturation=0.03125,
).start()

# If the image is STILL very red, uncomment the following line to manually reduce red gain:
# stream.picam2.set_controls({"AwbMode": 0, "ColourGains": (1.0, 2.0)})

app = create_app(stream)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)