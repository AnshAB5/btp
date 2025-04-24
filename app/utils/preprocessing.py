import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

# Image constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3  # RGB

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for VGG19 model
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array or None if loading fails
    """
    try:
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)  # VGG-19 preprocessing
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
        
def preprocess_uploaded_image(uploaded_file):
    """
    Process an uploaded file from Streamlit
    
    Args:
        uploaded_file: File object from st.file_uploader
        
    Returns:
        Preprocessed image array ready for the model
    """
    try:
        # Create a temporary file to save the uploaded image
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        # Preprocess the image
        img_array = load_and_preprocess_image(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        return None
