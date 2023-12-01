import torch
from torchvision import transforms as T
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
import streamlit as st
import requests
import io
import numpy as np


resize_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((940, 640)),
])


def make_prep(image):
    image = np.array(image)
    image_pil = Image.fromarray(image).convert('L')
    return resize_transform(np.array(image_pil))


def predict_1(image):

    image = make_prep(image)
    class ConvAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.SELU()
            )

            self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.SELU()
            )

            self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True)

            self.unpool = nn.MaxUnpool2d(2, 2)

            self.convtr1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.SELU()
            )

            self.convtr2 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )


        def encode(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x, ind = self.pool(x)

            return x, ind

        def decode(self, x, ind):
            x = self.unpool(x, ind)
            x = self.convtr1(x)
            x = self.convtr2(x)

            return x

        def forward(self, x):
            latent_x, ind = self.encode(x)
            x =  self.decode(latent_x, ind)

            return x

    
    device = torch.device('cpu')
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load('model/autoencoder.pt', map_location=device))
    model.eval()
    transform = T.Compose([
    T.ToTensor(),
])
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to(device).float() 
    model = model.to(device)

    with torch.no_grad():
        output_image = model(input_image)
    
    return output_image.squeeze().cpu().numpy()
