from fastapi import FastAPI, HTTPException
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import streamlit as st

transform_gray = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

transform_rgb = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class CheckImageVGGGrey(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


class CheckImageVGGRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

check_image_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gray= CheckImageVGGGrey()
model_rgb = CheckImageVGGRGB()
state_dict_gray = torch.load('model_gray_trash.pth', map_location=device)
model_gray.load_state_dict(state_dict_gray, strict=False)
model_gray.to(device)
model_gray.eval()
state_dict_rgb = torch.load('model_rgb_trash.pth', map_location=device)
model_rgb.load_state_dict(state_dict_rgb, strict=False)
model_rgb.to(device)
model_rgb.eval()


with st.sidebar:
    st.header('Menu')
    name = st.radio('Choose', ['grey', 'rgb'])

class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
if name == 'grey':
    st.title('Trash AI Classifier')
    st.text('Upload image with a number, and model will recognize it')

    file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg', 'webp'])

    if not file:
        st.warning('No file is uploaded')
    else:
        st.image(file, caption='Uploaded image')
        if st.button('Recognize the image'):
            try:
                image_data = file.read()
                if not image_data:
                    raise HTTPException(status_code=400, detail='No image is given')
                img = Image.open(io.BytesIO(image_data))
                img_tensor = transform_gray(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model_gray(img_tensor)
                    pred = y_pred.argmax(dim=1).item()

                st.success({'Prediction': class_names[pred]})

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

if name == 'rgb':
    st.title('Trash AI Classifier')
    st.text(f'Upload image with a number, and model will recognize it {class_names}')

    file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg', 'webp'])

    if not file:
        st.warning('No file is uploaded')
    else:
        st.image(file, caption='Uploaded image')
        if st.button('Recognize the image'):
            try:
                image_data = file.read()
                if not image_data:
                    raise HTTPException(status_code=400, detail='No image is given')
                img = Image.open(io.BytesIO(image_data))
                img_tensor = transform_rgb(img).unsqueeze(0).to(device)


                with torch.no_grad():
                    y_pred = model_rgb(img_tensor)
                    pred = y_pred.argmax(dim=1).item()

                st.success({'Prediction': class_names[pred]})

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))



