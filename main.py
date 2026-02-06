import cv2
import time
import plot
import time
import random
import avisengine
import numpy as np
from PIL import Image
from utils import utilss
from gan import generator, device
from object_detection import ObjectDetection
import pandas as pd

YOLO = ObjectDetection()
utils = utilss()

def update_values(*args):
        global rain, snow, haze
        rain = cv2.getTrackbarPos("Rain", "Adverse Condition")
        snow = cv2.getTrackbarPos("Snow", "Adverse Condition")
        haze = cv2.getTrackbarPos("Haze", "Adverse Condition")
        
def window(rain, snow, haze):
    cv2.namedWindow("Adverse Condition")
    cv2.resizeWindow("Adverse Condition", 400, 110)
    
    cv2.createTrackbar("Rain", "Adverse Condition", rain, 5, update_values)
    cv2.createTrackbar("Snow", "Adverse Condition", snow, 5, update_values)
    cv2.createTrackbar("Haze", "Adverse Condition", haze, 5, update_values)

rain, snow, haze = 5, 0, 5
window(rain, snow, haze)

kp1 = 20
ki1 = 10
kp2 = 30
ki3 = 0.01
kd = 0.05
l = 10
dt = 1

steer = 0

error_x = 0
error_steer = 0

counter = 0
current = 256
reference = 256

turn_current1 = np.linspace(256, 256 + 128, 80)
turn_current1 += np.arange(1, len(turn_current1) + 1) * 0.01

turn_current2 = np.linspace(256 + 100, 290, 230)
turn_current2 -= np.arange(1, len(turn_current2) + 1)[::-1] * 0.01

turn_current3 = np.linspace(280, 256, 180)
turn_current3 += np.arange(1, len(turn_current3) + 1) * 0.01

c = 1000
s = 0
t = time.time()

steerings, x, speeds, x_errors, ref, y = [], [], [], [], [], []

car = avisengine.Car()
car.connect("127.0.0.1", 25001)

car.setSteering(steer)
car.setSensorAngle(30)
i = 0
tm = time.time()

try:
    while(True):

        counter += 1
        car.getData()

        if(counter > 4):

            car.setSpeed(20)

            sensors = car.getSensors()
            main_frame = car.getImage()
            main_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)

            adverse_frame = utils.rain(main_frame, rain)
            adverse_frame = utils.snow(adverse_frame, snow)
            adverse_frame_ = utils.haze(adverse_frame, haze)

            adverse_frame = Image.fromarray(adverse_frame_)
            
            adverse_frame = utils.transform(adverse_frame)
            adverse_frame = adverse_frame.unsqueeze(0)
            normal_frame = generator(adverse_frame.to(device), 0).cpu().detach()
            normal_frame = cv2.cvtColor(utils.unnorm(normal_frame[0]).permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
            normal_frame = (normal_frame*255).astype(np.uint8)

            detected_frame, light, stop, crosswalk = YOLO.main_detector(normal_frame)

            if (sensors[1] < 630 - i*50) and (time.time()-t>25) and (i<2):
                print('obstacle detected!')
                ts = time.time()

                steerings, speeds, x, ref, x_errors, y = utils.turn_direction(-100, 3.1, car, turn_current1, 
                steerings, speeds, x, ref, x_errors, y, 5, 65, True)

                steerings, speeds, x, ref, x_errors, y = utils.turn_direction(100, 10.6, car, turn_current2,
                steerings, speeds, x, ref, x_errors, y, 5, 203, False)

                steerings, speeds, x, ref, x_errors, y = utils.turn_direction(-100, 6.25, car, turn_current3,
                steerings, speeds, x, ref, x_errors, y, 5, 122, False)

                print('turning took', time.time() - ts, 'seconds')

                print('back in the lane!')

                i+=1

                if i==2:
                    c = counter
             
            else:
                current = utils.getPath(normal_frame)
                error_x = (reference - current)
                steer = -kp2*current -ki3*current*dt-ki3*current*dt*dt-kd*current
                yaw = (car.getSpeed()/l)*steer
                steer = -kp1*yaw-ki1*yaw*dt
                 
            if i>0:
                car.setSteering(steer)

            plot.plot(steerings, speeds, x, ref, x_errors, y)

            cv2.imshow('Adverse Frame                                                                                                Normal Frame',
                       np.concatenate([cv2.resize(cv2.cvtColor(adverse_frame_, cv2.COLOR_RGB2BGR), (512,512)),
                                       cv2.resize(detected_frame, (512,512))],1))
            
            cv2.waitKey(1)

            if counter-c>3:
                break

            if cv2.waitKey(10) == ord('q'):
                break

            time.sleep(0.001)

finally:
    car.stop()