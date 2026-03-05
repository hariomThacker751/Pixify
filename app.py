import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

# Config
st.set_page_config(page_title="AI Detector", layout="wide")

# Load model ONCE
@st.cache_resource
def load_model():
    model = torch.jit.load('ai_detector_hybrid.pt', map_location='cpu')
    return model.eval()

model = load_model()

# FFT function
def get_fft(image):
    gray = np.array(image.convert('L'))
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])+1)
    return Image.fromarray(cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX).astype('uint8'))

# Preprocess
def preprocess(image):
    rgb_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    fft_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    rgb = rgb_tf(image).unsqueeze(0)
    fft_img = get_fft(image)
    fft = fft_tf(fft_img).unsqueeze(0)
    return rgb, fft

st.title("🔍 AI or Real?")

uploaded_file = st.file_uploader("Upload image", type=['png','jpg','jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original")
    with col2:
        st.image(get_fft(image), caption="FFT")
    
    if st.button("**PREDICT**", type="primary"):
        with st.spinner("Analyzing..."):
            rgb, fft = preprocess(image)
            prob = torch.sigmoid(model(rgb, fft)).item()
            
            st.markdown("### **RESULT**")
            if prob > 0.5:
                st.error("🤖 **AI GENERATED**")
                st.metric("AI Confidence", f"{prob*100:.0f}%")
            else:
                st.success("📸 **REAL**")
                st.metric("Real Confidence", f"{(1-prob)*100:.0f}%")
