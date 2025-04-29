# GAN_Erase-Cloud-from-Satellite-Images
# ☁️ Cloud Removal from Satellite Images using GAN (U-Net + ResNet)

This project uses a **Generative Adversarial Network (GAN)** architecture combining **U-Net** and **ResNet** components to effectively **remove clouds from satellite images**, enhancing the clarity and usability of remote sensing data for various applications like agriculture, disaster monitoring, and urban planning.

## 🚀 Project Overview

Clouds often obstruct ground details in satellite imagery, reducing their effectiveness for analysis. This project implements a **deep learning pipeline** that takes a cloudy satellite image as input and outputs a de-clouded version using a custom GAN model.

### Key Features:
- **U-Net Generator**: Captures spatial details via skip connections.
- **ResNet Blocks**: Improve feature learning and enhance the generator’s ability to restore fine details.
- **PatchGAN Discriminator**: Evaluates realism at the patch level for sharper results.

## 🧠 Model Architecture

- **Generator**:  
  - Built using a **U-Net encoder-decoder** backbone with embedded **ResNet blocks** for enhanced feature learning and residual refinement.
- **Discriminator**:  
  - Based on the **PatchGAN** concept, which classifies whether overlapping image patches are real or fake.
- **Loss Functions**:  
  - **Adversarial Loss (Binary Cross Entropy)**  
  - **L1 Loss (Pixel-wise similarity)**  
  - Optional: **Perceptual Loss** for high-level feature alignment

## 🗂 Dataset

The model is trained on a paired dataset containing:
- Cloudy images (input)
- Ground truth cloud-free images (target)
