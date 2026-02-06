from ultralytics import YOLO
import numpy as np
import cv2
import math 

class ObjectDetection:
    def __init__(self, ):

        self.model = YOLO('weights//yolov8n.pt')
        self.traffic_model = YOLO('weights//Yolov8TrafficLightColor.pt')
        self.crosswalk_model = YOLO('weights//YOLOv8CrossWalk.pt')

        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        
        self.traffic_class = ["green", "red", "yellow"]
        
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classNames), 3))
        self.traffic_COLORS = [(0,255,0),(0,0,255),(0,255,255)]

    def main_detector(self, frame):
        light = None
        crosswalk = None
        stop = False
        i = 0

        results = self.model(frame, verbose=False, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
            
                if self.classNames[cls] == "stop":
                    stop = True

                # if self.classNames[cls] == "traffic light":
                #     if i == 0:
                #         frame, crosswalk = self.check_cross(frame)
                #         i += 1
                #     frame, light = self.traffic_light(frame)
                #     frame, crosswalk = self.check_cross(frame)

                if self.classNames[cls] == 'car' or self.classNames[cls] == 'truck' or self.classNames[cls] == 'bus' or self.classNames[cls] == "traffic light":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS[cls], 2)
                    cv2.putText(frame, self.classNames[cls], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[cls], 2)

        return frame, light, stop, crosswalk

    def traffic_light(self, frame):
        light = None
        results = self.traffic_model(frame, verbose=False, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                light = self.traffic_class[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.traffic_COLORS[cls], 2)
                cv2.putText(frame, light, [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.traffic_COLORS[cls], 2)
        return frame, light

    def check_cross(self, frame):
        crosswalk = None
        cross = self.crosswalk_model(frame, verbose=False, stream=True)
        for r in cross:
            crosswalk = True
            boxes = r.boxes
            for box in boxes:
                x1_cross, y1_cross, x2_cross, y2_cross = box
                x1_cross, y1_cross, x2_cross, y2_cross = int(x1_cross), int(y1_cross), int(x2_cross), int(y2_cross)
                cv2.rectangle(frame, (x1_cross, y1_cross), (x2_cross, y2_cross), (255,0,255), 2)
                cv2.putText(frame, 'crosswalk', [x1_cross, y1_cross], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        return frame, crosswalk

    