import numpy as np
import cv2
import time
import pathlib
import sys

sys.path.insert(1, str(pathlib.Path.cwd().parents[0]) + "/common")

"""
a blueprint for a bounded box with its corresponding name,confidence score and 
"""
print(pathlib.Path.cwd())

class BoundedBox:
    def __init__(self, xmin, ymin, xmax, ymax, ids, confidence):
        with open(str(pathlib.Path.cwd().parents[0]) + "/datas/coco.names", 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')  # List of classes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.name = self.classes[ids]
        self.confidence = confidence

"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""

class Lanes:
    def __init__(self, lanes):
        self.lanes = lanes

    def getLanes(self):
        return self.lanes

    def lanesTurn(self):
        return self.lanes.pop(0)

    def enque(self, lane):
        self.lanes.append(lane)

    def lastLane(self):
        return self.lanes[-1]

"""
a blueprint that has lanes as lists and give queue like functionality 
to reorder lanes based on their turn for green and red light state
"""
class Lane:
    def __init__(self, count, frame, lane_number):
        self.count = count
        self.frame = frame
        self.lane_number = lane_number

"""
given lanes object return a duration based on comparison of each lane vehicle count
"""
def schedule(lanes):
    standard = 10  # Standard duration
    reward = 0  # Reward to adjust the duration
    turn = lanes.lanesTurn()

    for i, lane in enumerate(lanes.getLanes()):
        if i == (len(lanes.getLanes()) - 1):
            reward += (turn.count - lane.count) * 0.2
        else:
            reward += (turn.count - lane.count) * 0.5

    scheduled_time = round((standard + reward), 0)
    lanes.enque(turn)
    return scheduled_time

"""
given duration and lanes, returns a grid image containing frames of each lane with
their corresponding waiting duration
"""   
def display_result(wait_time, lanes):
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)

    lane_frames = []
    for i, lane in enumerate(lanes.getLanes()):
        lane.frame = cv2.resize(lane.frame, (640, 360))  # Resize for concatenation

        if wait_time <= 0 and (i == (len(lanes.getLanes()) - 1) or i == 0):
            color = yellow
            text = "yellow: 2 sec"
        elif wait_time >= 0 and i == (len(lanes.getLanes()) - 1):
            color = green
            text = f"green: {wait_time} sec"
        else:
            color = red
            text = f"red: {wait_time} sec"

        lane.frame = cv2.putText(lane.frame, text, (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        lane.frame = cv2.putText(lane.frame, f"vehicle count: {lane.count}", (60, 195), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        lane_frames.append(lane.frame)

    # Concatenate lane frames into a grid
    top_row = np.concatenate(lane_frames[:2], axis=1)
    bottom_row = np.concatenate(lane_frames[2:], axis=1)
    all_lanes_image = np.concatenate((top_row, bottom_row), axis=0)

    return all_lanes_image

# given detected boxes, return number of vehicles on each box
def vehicle_count(boxes):
    vehicle = 0
    for box in boxes:
        if box.name in ["car", "truck", "bus"]:
            vehicle += 1
    return vehicle

# given the grid dimension, returns a 2d grid
def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def drawPred(frame, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=2)
    return frame

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ratioh, ratiow = frameHeight / 320, frameWidth / 320

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            if scores.size == 0:  # Check if scores is empty
                continue
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5 and detection[4] > 0.5:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    correct_boxes = []

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box
            box = BoundedBox(left, top, left + width, top + height, classIds[i], confidences[i])
            correct_boxes.append(box)
            frame = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

    return correct_boxes, frame

"""
interpret the output boxes into the appropriate bounding boxes based on the yolo paper 
logspace transform
"""
def modify(outs):
    # Simplified for compatibility
    return outs

"""
given each lanes image, it inferences onnx model on the image, return lanes object
containing processed image and waiting duration for each image
"""
def final_output(net, output_layer, lanes):
    for lane in lanes.getLanes():
        lane.frame = cv2.resize(lane.frame, (1280, 720))
        blob = cv2.dnn.blobFromImage(lane.frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(output_layer)
        dets = modify(layerOutputs)
        boxes, frame = postprocess(lane.frame, dets)
        lane.count = vehicle_count(boxes)
        lane.frame = frame
    return lanes

# Define wait_time before using it
wait_time = 0  # Initialize wait_time with a default value

start = time.time()
# Initialize net (ONNX model or other network)
net = cv2.dnn.readNetFromONNX("d:\\projects\\review\\AI-based-Traffic-Control-System--\\models\\yolov5s.onnx")  # Update with the correct path to your ONNX model

# Initialize ln (output layer names)
ln = net.getUnconnectedOutLayersNames()

# Initialize lanes (example with dummy data)
lane1 = Lane(count=0, frame=np.zeros((720, 1280, 3), dtype=np.uint8), lane_number=1)
lane2 = Lane(count=0, frame=np.zeros((720, 1280, 3), dtype=np.uint8), lane_number=2)
lane3 = Lane(count=0, frame=np.zeros((720, 1280, 3), dtype=np.uint8), lane_number=3)
lane4 = Lane(count=0, frame=np.zeros((720, 1280, 3), dtype=np.uint8), lane_number=4)
lanes = Lanes([lane1, lane2, lane3, lane4])

lanes = final_output(net, ln, lanes)  # Returns lanes object with processed frame
print("Processed lanes:", lanes)  # Debugging output
end = time.time()
print("Total processing time: " + str(end - start))

if wait_time <= 0:
    images_transition = display_result(wait_time, lanes)
    print("Images transition:", images_transition)  # Debugging output
    final_image = cv2.resize(images_transition, (1020, 720))
    cv2.imshow("f", final_image)
    cv2.waitKey(100)

    wait_time = schedule(lanes)  # Returns waiting duration of each lane
    print("Wait time:", wait_time)  # Debugging output
