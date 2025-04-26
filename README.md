# Chatter Detection App (Image & Video Classification)

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit Version](https://img.shields.io/badge/Streamlit-1.31.0-orange.svg)](https://streamlit.io)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This Streamlit application provides a user interface for classifying **images** and **video frames** to detect "Chatter" or "Non Chatter" states, likely related to a manufacturing or machining process.

It utilizes a multi-input Deep Learning model based on VGG19 for visual feature extraction, combined with numerical process parameters (e.g., Depth of cut, RPM) for a more context-aware classification.

---

## Features

* **Dual Mode Operation:** Switch between Image and Video processing.
* **Image Classification:** Upload individual images for classification.
* **Video Classification:** Upload videos; the app processes each frame individually.
* **Multi-Input Model:** Leverages both visual data and user-provided numerical features.
* **Numerical Feature Input:** Sidebar for entering process parameters (e.g., Depth of cut, RPM).
* **Real-time Feedback (Video):** View classification results per frame as you navigate the video timeline.
* **Probability Scores:** Displays the model's confidence score for the "Chatter" class.
* **Interactive UI:** Built with Streamlit for ease of use.

---

## Technology Stack

* **Backend/ML:** Python, TensorFlow (Keras API), OpenCV
* **Frontend/UI:** Streamlit
* **Core Libraries:** Pandas, NumPy, Pillow

---

## Project Structure

```
.
├── app/                          # Main application directory
│   ├── app.py                    # The main Streamlit application script
│   ├── requirements.txt          # Python package dependencies
│   ├── utils/                    # Utility functions
│   │   └── preprocessing.py      # Image/frame preprocessing logic
│   └── models/                   # Directory to store the trained model
│       └── vgg19_chatter_multi_input_model.keras
├── .streamlit/                   # (Optional) Streamlit configuration
│   └── secrets.toml              # Secrets and config overrides
├── README.md                     # Project overview and instructions
└── .gitignore                    # Files and directories to ignore in Git
```

---

## Setup Instructions

Follow these steps to set up the project locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/chatter-detection-app.git
   cd chatter-detection-app
   ```

2. **Create a virtual environment (highly recommended):**

   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r app/requirements.txt
   ```

4. **Run the Streamlit application:**

   ```bash
   streamlit run app/app.py
   ```

The application should open automatically in your web browser.

---

## Usage

1. **Select Mode:** Choose either "Image Processing" or "Video Processing" using the radio buttons.
2. **Enter Numerical Features:** Use the sidebar to input the required numerical values (e.g., Depth of cut, RPM). These values are normalized internally before being fed to the model.
3. **Upload File:**
    - **Image Mode:** Upload a single image file (JPG, PNG, etc.). Click "Run Image Prediction".
    - **Video Mode:** Upload a video file (MP4, AVI, etc.). Click "Process Video Frames". Processing may take time depending on video length.
4. **View Results:**
    - **Image Mode:** The classification ("Chatter" / "Non Chatter") and probability score will be displayed.
    - **Video Mode:** After processing, the video player and a timeline slider will appear. Navigate the slider to see the classification for the frame closest to that timestamp.

---

## Configuration & Customization

* **Model Path:** Modify the `model_path` variable in `app.py` if your model file has a different name or location.
* **Numerical Features & Normalization:** Adjust the feature names, input ranges (`st.sidebar.number_input`), and the `mean_values`, `std_values` arrays in `app.py` to match the features and normalization used during your model training.
* **Prediction Threshold:** Change the `PREDICTION_THRESHOLD` constant in `app.py` (default is 0.5) to tune the classification boundary based on your model's performance (e.g., using a ROC curve analysis).
* **Preprocessing:** Modify image/frame preprocessing steps (resizing, normalization method) in `utils/preprocessing.py` and `app.py` (`preprocess_frame` function) if your model requires different input processing. Ensure both image and video preprocessing are consistent.

---

## Deployment Notes (Streamlit Community Cloud)

* This app can be deployed using Streamlit Community Cloud.
* **Important:** During deployment setup, go to "Advanced settings" and select **Python 3.9** as the Python version.
* Ensure your `requirements.txt` file is correctly placed in the repository and lists all necessary packages (including `tensorflow==2.15.0` and `opencv-python-headless`).

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

