import torch
import streamlit as st
from src.config import MODEL_PATH

@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location='cpu')
    return model.eval()
