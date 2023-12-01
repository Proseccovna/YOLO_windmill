from ultralytics import YOLO
import torch
import PIL
import os
import matplotlib.pyplot as plt
import cv2


model = YOLO('yolo_model/best_of_the_best.pt')


def detect(image):

    img = cv2.imread(image) 
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

    return detect_img

    