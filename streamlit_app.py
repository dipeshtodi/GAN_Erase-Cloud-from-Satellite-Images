import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from Unet_model import UNet

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Title
st.title("☁️ Satellite Image Cloud Removal GAN")

uploaded_files = st.file_uploader("Upload Satellite Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)[0].cpu().numpy()

        # Display
        st.subheader("Original vs Cleaned")
        col1, col2 = st.columns(2)

        col1.image(image, caption="Input", use_column_width=True)
        col2.image(np.transpose(output, (1, 2, 0)), caption="Generated", use_column_width=True)
