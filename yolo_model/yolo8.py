from ultralytics import YOLO
import torch
import PIL
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = YOLO('yolo_model/best2.pt')


def detect(image):


    img = Image.open(image).convert("RGB")
    img_array = np.array(img)
    detect_result = model(img_array)
    detect_img = detect_result[0].plot()
    detect_img = Image.fromarray(detect_img)

    return detect_img



    