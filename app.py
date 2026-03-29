import streamlit as st
import torch
import time
from PIL import Image

from src.model_loader import load_model
from src.preprocessor import preprocess, get_fft

# Page Config
st.set_page_config(page_title="Pixify | AI Detector", page_icon="✨", layout="centered")

# Professional UI/UX CSS Injection
st.markdown("""
<style>
    /* Global Reset & Base */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 800px; /* Keep it centered and focused */
    }
    
    /* Hero Header */
    .hero {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    .hero-title {
        font-weight: 800;
        font-size: 4rem;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: Layout
    }
    .hero-subtitle {
        color: #718096;
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* Analyze Button Override */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        color: white;
        border: none;
        box-shadow: 0 10px 20px -10px rgba(236, 72, 153, 0.5);
        border-radius: 12px;
        height: 3.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px -10px rgba(236, 72, 153, 0.8);
    }
    
    /* Image Styling */
    .image-preview img {
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        object-fit: contain;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Pixify</h1>
    <p class="hero-subtitle">Deepfake & AI Generation Detection</p>
</div>
""", unsafe_allow_html=True)

# Main Application Flow
uploaded_file = st.file_uploader("Upload an image to verify...", type=['png','jpg','jpeg'], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Image Display
    st.markdown("### Image Preview")
    colA, colB, colC = st.columns([1, 4, 1])
    with colB:
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analysis Trigger
    if st.button("Run Deep Analysis"):
        st.markdown('</div>', unsafe_allow_html=True) # close top card
        
        # Simulated Progress for UX
        progress_text = "Initializing neural engine..."
        my_bar = st.progress(0, text=progress_text)
        
        # Load Model
        model = load_model()
        my_bar.progress(30, text="Extracting spatial features...")
        time.sleep(0.3)
        
        # Preprocess & Infer
        rgb, fft = preprocess(image)
        my_bar.progress(60, text="Analyzing frequency domain (FFT)...")
        prob = torch.sigmoid(model(rgb, fft)).item()
        time.sleep(0.3)
        my_bar.progress(100, text="Analysis complete!")
        time.sleep(0.2)
        my_bar.empty()
        
        # Results Section
        st.markdown("---")
        
        if prob > 0.5:
            st.error("🚨 This image exhibits strong signs of AI generation.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="AI Probability", value=f"{prob*100:.1f}%", delta="Generated", delta_color="inverse")
            with col2:
                st.metric(label="Authenticity", value=f"{(1-prob)*100:.1f}%")
        else:
            st.success("✅ This image appears to be authentic and completely real.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Authentic Probability", value=f"{(1-prob)*100:.1f}%", delta="Real", delta_color="normal")
            with col2:
                st.metric(label="AI Confidence", value=f"{prob*100:.1f}%")
        
        # Technical Readout
        with st.expander("View Technical Analysis Data", expanded=False):
            st.markdown("### Frequency Domain Map")
            st.info("AI models often leave distinct, grid-like artifacts in the frequency domain that are invisible to the naked eye. This heatmap visualizes those patterns.")
            
            fft_heatmap = get_fft(image)
            st.image(fft_heatmap, use_container_width=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True) # close top card around image if not clicked

