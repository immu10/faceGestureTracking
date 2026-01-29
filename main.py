import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pprint
import math
from typing import Tuple, Union

model_path = r"I:\codebs\MediapipeModel\face_landmarker.task"
image_path = r"Screenshot 2025-10-27 000158.png"
img = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
new_frame = None
catMouthOpenFrame = cv2.imread("pics\catOpenMouth.png")
raisedEyebrowsFrame = cv2.imread(r"pics\neuronActivation.jpg")
def visualize(result,output_image: mp.Image,timestamp_ms: int):
    global detection_result,new_frame
    new_frame = output_image
    # height, width, _ = image.shape
    detection_result = result
    # print(result)
    

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,running_mode=vision.RunningMode.LIVE_STREAM, result_callback=visualize)
detector = vision.FaceLandmarker.create_from_options(options)
# frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_rate = 30
frame_idx = 0
cv2.namedWindow("name", cv2.WINDOW_NORMAL)
cv2.namedWindow("input", cv2.WINDOW_NORMAL)


while True:
    
    ret, frame = cap.read()
    # print("checkpoint 1")
    timestamp = frame_idx * (1000 / frame_rate)
    if ret:
        cv2.imshow("input",frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # didn't need
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # print("checkpoint")
        detector.detect_async(mp_image,int(timestamp))
        # print("checkpoint 2")

        if new_frame is not None and detection_result is not None and detection_result.face_landmarks:
            for detection in detection_result.face_landmarks:
                # print(detection,"2\n")
                global start_point,end_point
                # print(detection,"detection\n")              
                head_size = math.dist((detection[10].x,detection[10].y),(detection[152].x,detection[152].y))
                if head_size <0.3:
                    print("too far away, please come closer")
                    continue
                if head_size >0.45:
                    print("too close, please move back")
                    continue
                print(head_size,"head_size\n")
                
                raised_eyebrow_distance = math.dist((detection[107].x,detection[107].y),(detection[109].x,detection[109].y))
                # print(raised_eyebrow_distance,"raised_eyebrow_distance\n")
                if raised_eyebrow_distance < 0.05:
                    frame = raisedEyebrowsFrame.copy()
                    continue

                mouth_width = math.dist((detection[13].x,detection[13].y),(detection[14].x,detection[14].y))
                if mouth_width > 0.03:
                    frame = catMouthOpenFrame.copy()
                    continue

                
                for points in detection:
                    x,y = points.x * frame.shape[1], points.y * frame.shape[0]
                    cv2.circle(frame, (int(x),int(y)), 1, (0, 255, 0), -1)
                # Draw bounding box
                # else:
                #     frame = catMouthOpenFrame.copy()

                            
            
            cv2.imshow("name", frame)
            new_frame = None
        
        # # cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # # detection_result = detector.detect(image)
        frame_idx += 1





