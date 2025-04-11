import sys
import argparse
import pathlib
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template
import os

# Hide the first window
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

# Add the "common" directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import utils as util

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")  # Updated template folder path

def load_video(video_path):
    """Loads a video file and returns a VideoCapture object."""
    if not pathlib.Path(video_path).exists():
        raise FileNotFoundError(f"❌ Video file not found: {video_path}")
    return cv2.VideoCapture(video_path)

def load_model(model_path):
    """Loads the YOLOv5 ONNX model."""
    if not pathlib.Path(model_path).exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    net = cv2.dnn.readNet(model_path)
    
    # Use CPU for inference
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return net

def final_output(net, output_layer, lanes):
    for lane in lanes.getLanes():
        lane.frame = cv2.resize(lane.frame, (640, 640))  # Ensure correct input size
        blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(output_layer)
        dets = modify(layerOutputs)
        boxes, frame = postprocess(lane.frame, dets)
        lane.count = vehicle_count(boxes)
        lane.frame = frame
    return lanes

def process_frames():
    """Generator function to process and yield video frames."""
    global cap_list, net, ln, lanes
    wait_time = 0

    while True:
        frames = []
        
        # Read frames from each video
        for cap in cap_list:
            ret, frame = cap.read()
            if not ret:
                # Restart the video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames.append(frame)
        
        for lane, frame in zip(lanes.getLanes(), frames):
            lane.frame = frame
        
        lanes = util.final_output(net, ln, lanes)
        
        if wait_time <= 0:
            transition_image = util.display_result(wait_time, lanes)
            final_image = cv2.resize(transition_image, (1020, 720))
            wait_time = util.schedule(lanes)
        else:
            scheduled_image = util.display_result(wait_time, lanes)
            final_image = cv2.resize(scheduled_image, (1020, 720))
            wait_time -= 1

        _, buffer = cv2.imencode('.jpg', final_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route to stream video frames."""
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Route for the dashboard."""
    return render_template('dashboard.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Control using YOLOv5 ONNX model")
    parser.add_argument("--sources", type=str, default="video1.mp4,video2.mp4,video3.mp4,video5.mp4")
    args = parser.parse_args()

    sources = args.sources.split(",")
    video_paths = [f"d:\\projects\\review\\AI-based-Traffic-Control-System--\\datas\\{src}" for src in sources]

    # Load videos
    cap_list = [load_video(video) for video in video_paths]

    # Load YOLOv5 ONNX model
    model_path = r"d:\\projects\\review\\AI-based-Traffic-Control-System--\\models\\yolov5s.onnx"
    net = load_model(model_path)
    ln = net.getUnconnectedOutLayersNames()

    # Initialize lanes
    lanes = util.Lanes([util.Lane(0, None, i + 1) for i in range(4)])

    # Ensure OpenCV windows are hidden
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)