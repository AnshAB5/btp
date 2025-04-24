# Image Classification Streamlit App

This application provides a user interface for classifying images using a VGG19-based model that also incorporates numerical features.

## Setup Instructions

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place your model file in the `models/` directory:
   - Ensure the model is named `vgg19_chatter_multi_input_model.keras`
   - Or update the model path in `app.py`

4. Run the application:
   ```
   streamlit run app.py
   ```

## Features

- Upload and classify images
- Input numerical features through the sidebar
- View classification results with probability scores
- Simple and intuitive UI

## Customization

- Adjust the numerical features in `app.py` to match your model's requirements
- Modify the preprocessing in `utils/preprocessing.py` if needed
- Change the classification threshold in `app.py` based on your model's optimal point

## Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- Pandas
- NumPy
- Pillow

## Notes

- The app is designed for local use
- All processing happens on your machine
- No data is sent to external servers
