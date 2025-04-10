import sys
import argparse
import pathlib
import time
import cv2
import numpy as np

# Add the "common" directory to the Python path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import utils as util

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

def main(sources):
    """Main function for processing traffic control."""
    video_paths = [f"d:\\projects\\review\\AI-based-Traffic-Control-System--\\datas\\{src}" for src in sources]
    
    # Load videos
    cap_list = [load_video(video) for video in video_paths]
    
    # Load YOLOv5 ONNX model
    model_path = r"d:\\projects\\review\\AI-based-Traffic-Control-System--\\models\\yolov5s.onnx"
    net = load_model(model_path)
    ln = net.getUnconnectedOutLayersNames()
    
    # Initialize lanes
    lanes = util.Lanes([util.Lane(0, None, i + 1) for i in range(4)])
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
            cv2.imshow("Traffic Control", final_image)
            wait_time = util.schedule(lanes)
        else:
            scheduled_image = util.display_result(wait_time, lanes)
            final_image = cv2.resize(scheduled_image, (1020, 720))
            cv2.imshow("Traffic Control", final_image)
            wait_time -= 1

        # Wait for a key press and allow quitting with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    for cap in cap_list:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Control using YOLOv5 ONNX model")
    parser.add_argument("--sources", type=str, default="video1.mp4,video2.mp4,video3.mp4,video5.mp4")  # Updated to remove video4.mp4
    args = parser.parse_args()

    sources = args.sources.split(",")
    main(sources)