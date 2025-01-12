import os
import cv2
from IPython.display import display, Image
import ipywidgets as widgets
import threading
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera

# ====================

# 카메라 레코딩을 스레드
def view():
    frame = camera.read()
    frame_index = 0

    while True:
        if video:
            input("Record")
            frame = camera.read()
            if frame is not None:
                image_filename = os.path.join("image", f"frame_{frame_index:09d}.jpg")
                cv2.imwrite(image_filename, frame[:,:,::-1])  # OpenCV는 BGR 형식을 사용하므로 RGB로 변환하여 저장
                print(f"Image saved as {image_filename}")
                frame_index += 1

# ====================

# 스레드 하나 열어서, 조이스틱 버튼 누르면 젯레이서 카메라 저장
video = True
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
thread = threading.Thread(target=view, args=())
thread.start()