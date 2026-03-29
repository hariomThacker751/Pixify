# Pixify

AI-powered deepfake and synthetic image detection tool. Pixify analyzes images to determine whether they are authentic or generated using artificial intelligence, with emphasis on frequency domain analysis to identify hidden artifacts.

## Overview

Pixify uses a dual-stream neural network architecture that processes images in both spatial and frequency domains. By analyzing the FFT (Fast Fourier Transform) representation of images, the model can detect subtle artifacts that AI generators leave behind, patterns often invisible to human eyes.

The application provides an interactive web interface built with Streamlit, making it accessible to both technical and non-technical users who need to verify image authenticity.

## Features

- Real-time image analysis for deepfake and AI-generated content detection
- Dual-stream processing combining spatial and frequency domain analysis
- Interactive web interface with progress tracking
- Detailed probability metrics for authenticity assessment
- FFT visualization showing frequency domain artifacts
- Support for common image formats (PNG, JPG, JPEG)
- GPU acceleration with PyTorch

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- CUDA-capable GPU (optional but recommended for faster inference)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/hariomThacker751/Pixify.git
cd Pixify
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`

### How to Use

1. **Upload Image**: Click the upload area or drag-and-drop an image you want to analyze
2. **Image Preview**: Review the uploaded image to ensure it's the correct one
3. **Run Analysis**: Click the "Run Deep Analysis" button to begin detection
4. **View Results**: The system will display:
   - Detection result (AI-generated or authentic)
   - Confidence percentage
   - Authenticity probability
5. **Technical Details**: Expand the "View Technical Analysis Data" section to see the FFT heatmap

### Supported Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- Maximum recommended image size: 2048x2048 pixels

## Project Structure

```
Pixify/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py         # Configuration settings
│   ├── model_loader.py   # Model loading utilities
│   ├── preprocessor.py   # Image preprocessing and FFT analysis
│   └── __init__.py
├── assets/               # Images and resources
└── results/              # Analysis output directory
```

## Technical Details

### Architecture

The detection model uses a two-branch architecture:

- **Spatial Branch**: Processes raw RGB image features using convolutional networks
- **Frequency Branch**: Analyzes FFT-transformed image to detect generation artifacts

Both branches are fused to produce a final confidence score for whether an image is AI-generated.

### How It Works

AI generation models often produce distinctive patterns in the frequency domain. These patterns result from:

- Repeated convolution operations in generator networks
- Quantization artifacts from the generation process
- Specific architectural signatures of different AI models

By examining the FFT representation, Pixify can identify these subtle fingerprints that betray synthetic content.

## Dependencies

- **streamlit** (1.38.0): Web application framework
- **torch** (2.4.1): Deep learning framework
- **torchvision** (0.19.1): Computer vision utilities
- **opencv-python-headless** (4.10.0.84): Image processing
- **pillow** (10.4.0): Image handling
- **numpy** (1.26.4): Numerical computations

## Performance Notes

- First run will download the pre-trained model (~500MB)
- GPU acceleration significantly improves analysis speed
- CPU inference typically takes 5-10 seconds per image
- GPU inference typically takes 1-2 seconds per image

## Model Details

The detection model has been trained on a diverse dataset of both authentic images and AI-generated content from various sources and generators. The model achieves high accuracy across different image types and resolutions.

## Limitations

- Model performance may vary with heavily compressed or watermarked images
- Very small images (< 128x128) may produce less reliable results
- Generated content from emerging AI models not seen during training may have lower detection accuracy

## Output Files

- Analysis results and images are stored in the `results/` directory
- FFT heatmaps are generated on-demand during analysis

## Development

To modify or extend this project:

1. Edit model configuration in `src/config.py`
2. Update preprocessing logic in `src/preprocessor.py`
3. Load custom models in `src/model_loader.py`
4. Modify the UI in `app.py`

## Troubleshooting

**Port already in use**: Change the port with `streamlit run app.py --server.port 8502`

**Out of memory errors**: Reduce image size or use `--logger.level=debug` for diagnostics

**Model loading fails**: Ensure you have sufficient disk space and internet connectivity

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

Built with Streamlit and PyTorch. Last updated March 2026.