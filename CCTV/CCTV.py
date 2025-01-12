from cnn.center_dataset import TEST_TRANSFORMS
import os
import cv2
import ipywidgets as widgets
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera
from time import sleep
import torch
import torchvision
import numpy as np
from ultralytics import YOLO
import websockets
import asyncio

# ====================

##### 클라이언트 비동기함수 #####
async def code():
    
    ## Connection ##
    uri = "ws://192.168.0.206:8000"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Camera")
        connection = await websocket.recv()
        print("From Server :", connection)

    ## Detection ##
        while(True):

            # Camera
            frame = camera.read()
            if frame is not None:
                capture_filename = os.path.join("capture", f"capture.jpg")
                cv2.imwrite(capture_filename, frame[:,:,::-1])

            # Model
            result = model_number.predict(source='capture', save=True)   
            label = result[0].__dict__['boxes'].cls  
            location = result[0].__dict__['boxes'].xyxy
            number = ""
            if(label.size(0) == 2):
                if(location[0][0] <= location[1][0]):
                    number = str(int(label[0]+1)) + str(int(label[1]+1))
                elif(location[0][0] > location[1][0]):
                    number = str(int(label[1]+1)) + str(int(label[0]+1))
            result = model_situation.predict(source='capture', save=True)   
            label = result[0].__dict__['boxes'].cls
            situation = ""
            if(label.size(0) == 1):
                if(label[0] == 0):
                    situation = "Construction" 
                elif(label[0] == 1):
                    situation = "Pedestrian"
            
            # Send
            message = number + "/" + situation
            await websocket.send(message)
            await websocket.recv()

# ====================

##### 클라이언트 열기 #####

# Set Camera 
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

# Set Model
    ##### Number #####
model_number = YOLO('Number_Model.pt', task='detect')
device = torch.device('cuda')
model_number = model_number.to(device)
    ##### Situation #####
model_situation = YOLO('Situation_Model.pt', task='detect')
device = torch.device('cuda')
model_situation = model_situation.to(device)

# Set Client
asyncio.get_event_loop().run_until_complete(code())