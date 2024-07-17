import io
from bsrgan import BSRGAN
import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import base64

# Function to compress image
def compress_image(im, quality=80):
    try:
        # Convert to RGB (if image is not already in RGB mode)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        
        # Create an in-memory stream for holding the compressed image
        output = io.BytesIO()
        
        # Save the image to the in-memory stream with JPEG format and given quality
        im.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
        
        # Get the compressed image data as bytes
        compressed_image_bytes = output.getvalue()
        
        # Close the in-memory stream
        output.close()
        
        return compressed_image_bytes
    
    except Exception as e:
        st.write("Error compressing image:", e)
        return None

# Function to create a download link for a file
def get_binary_file_downloader_html(bin_file, file_label='File', button_label='Download'):
    with io.BytesIO(bin_file) as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">{button_label}</a>'
    return href

# Streamlit UI
st.markdown("<h2 style='text-align: center;'> IMAGE ENHANCEMENT </h2>", unsafe_allow_html=True)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize BSRGAN model
model_bsrgan = BSRGAN(model_path='kadirnar/bsrgan', device=device, hf_model=True)
model_bsrgan.save = True

# File upload UI
image = st.file_uploader(label='Upload Image', accept_multiple_files=False, key='fileUploader')

# Button to trigger enhancement
submit = st.button('Enhance Image')

# When the button is clicked
if submit and image is not None:
    # Read and convert the uploaded image to bytes
    image_bytes = image.read()
    
    # Ensure image is properly read and decoded
    try:
        # Convert image bytes to NumPy array for processing with OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        degraded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        
        # Check if image size is sufficient for enhancement
        if degraded.shape[0] > 1440:
            st.write('Image quality is good enough, no need to enhance it')
        else:
            # Temporarily save the degraded image for prediction
            temp_path = 'temp_degraded.png'
            cv2.imwrite(temp_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
            
            # Predict using BSRGAN model
            pred = model_bsrgan.predict(temp_path)
            
            # Assuming pred is a path to the enhanced image, open and compress the image
            if isinstance(pred, str) and pred.endswith('.png'):
                pred_image = Image.open(pred)
                enhanced_image_bytes = compress_image(pred_image)
                
                if enhanced_image_bytes is not None:
                    # Display original and enhanced images
                    st.image([degraded, enhanced_image_bytes], caption=['Original Image', 'Enhanced Image'], width=300)
                    
                    # Add a download button for the enhanced image
                    st.markdown(get_binary_file_downloader_html(enhanced_image_bytes, 'Enhanced_Image.jpg', 'Download Enhanced Image'), unsafe_allow_html=True)
    
    except Exception as e:
        st.write("Error processing image:", e)


