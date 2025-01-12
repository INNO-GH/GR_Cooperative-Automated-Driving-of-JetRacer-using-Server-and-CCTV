from cnn.center_dataset import TEST_TRANSFORMS
from jetracer.nvidia_racecar import NvidiaRacecar
import os
import cv2
from datetime import datetime
import pygame
from IPython.display import display, Image
import ipywidgets as widgets
import threading
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera
from time import sleep
import torch
import torchvision
import PIL.Image
import copy
import numpy as np
from ultralytics import YOLO
import re
import time
import websockets
import asyncio

# ====================

# Function Get Model
def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

# Function PreProcess
def preprocess(capture: PIL.Image):
    device = torch.device('cuda')    
    capture = TEST_TRANSFORMS(capture).to(device)
    return capture[None, ...]

# Function PD Controller
Kp = 1.0
Kd = 0.05
dt = 0.1
pre_error = 0
def PDController(error):
    global pre_error
    P = Kp * error
    derivative = (error - pre_error) / dt
    D = Kd * derivative
    steering = (P - D) * 0.0025
    steering = max(min(steering, 1), -1)
    pre_error = error
    return steering

# ====================

# Communication and Linetracing
async def code():
    
    ##### Connect #####
    global Mode, Lock, ThrottleNormal
    uri = "ws://192.168.0.206:8000"
    async with websockets.connect(uri) as websocket:
        await websocket.send("25")
        connection = await websocket.recv()
        print("From Server :", connection)

    ##### Driving #####
        while(True):
            
            ##### Mode #####
            if(Lock == 0):
                await websocket.send("")
                sign = await websocket.recv()
                if(sign != ""):
                    print("From Server :", sign)
                    if(sign == "Pedestrian"):
                        car.throttle = 0
                        sleep(3)
                        continue
                    elif(sign == "Construction"):
                        car.throttle = 0
                        sleep(3)
                        Lock = 100

            ##### Pause #####
            Lock = max(0, Lock-1)
            pygame.event.pump()
            if Mode==0:
        
                # JoyStick
                if joystick.get_button(10):
                    Mode = 1
                    continue

                # JetRacer
                car.throttle = 0
                car.steering = 0

            ##### Move #####
            elif Mode==1:
    
                # JoyStick
                if joystick.get_button(11):
                    Mode = 0
                    continue
                if joystick.get_button(6):
                    ThrottleNormal = ThrottleNormal + 0.0025
                if joystick.get_button(7):
                    ThrottleNormal = ThrottleNormal - 0.0025

                # Camera
                frame = camera.read()
                if frame is not None:
                    capture_filename = os.path.join("capture", f"capture.jpg")
                    cv2.imwrite(capture_filename, frame[:,:,::-1])

                # Model
                capture_filename_fmt = 'capture/capture.jpg'
                capture_ori = PIL.Image.open(capture_filename_fmt)
                width = capture_ori.width
                height = capture_ori.height
                with torch.no_grad():
                    capture = preprocess(capture_ori)
                    if(Lock == 0):
                        output = model_left(capture).detach().cpu().numpy()
                    else:
                        output = model_right(capture).detach().cpu().numpy()
                x, y = output[0]
                x = (x / 2 + 0.5) * width
                y = (y / 2 + 0.5) * height

                # JetRacer
                car.steering = PDController(x-320)
                if(car.throttle == 0):
                    car.steering = 0
                    car.throttle = ThrottleNormal/2
                    car.throttle = ThrottleNormal 
                    car.throttle = ThrottleNormal+0.2
                    car.throttle = ThrottleNormal+0.4
                    car.throttle = ThrottleNormal+0.6
                    car.throttle = ThrottleNormal+0.8
                    car.throttle = ThrottleNormal+1.0
                    car.throttle = ThrottleNormal+1.2
                    car.throttle = ThrottleNormal+1.4
                    car.throttle = ThrottleNormal+1.2
                    car.throttle = ThrottleNormal+1.0
                    car.throttle = ThrottleNormal+0.8
                    car.throttle = ThrottleNormal+0.6
                    car.throttle = ThrottleNormal+0.4
                    car.throttle = ThrottleNormal+0.2
                    car.throttle = ThrottleNormal   
                    sleep(0.78)                  
                else:
                    car.throttle = ThrottleNormal

# ====================

# Set JoyStick
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
Mode = 0
Lock = 0

# Set JetRacer
car = NvidiaRacecar()
car.steering_offset = 0.08
car.throttle_gain = 0.5
ThrottleNormal = 0.26

# Set Camera 
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

# Set Model
    ##### Left #####
model_left = get_model()
model_left.load_state_dict(torch.load('Follow_Model_Left.pth'))
device = torch.device('cuda')
model_left = model_left.to(device)
    ##### Right #####
model_right = get_model()
model_right.load_state_dict(torch.load('Follow_Model_Right.pth'))
device = torch.device('cuda')
model_right = model_right.to(device)

# Set Client
asyncio.get_event_loop().run_until_complete(code())