import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

def get_fft(image):
    gray = np.array(image.convert('L'))
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])+1)
    return Image.fromarray(cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX).astype('uint8'))

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
