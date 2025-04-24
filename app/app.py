import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
# Assuming utils.preprocessing has the necessary functions
from utils.preprocessing import preprocess_uploaded_image, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
import warnings
import cv2 # Added for video processing
from io import BytesIO # Added for handling bytes
import tempfile # Added for temporary file handling
import time # Added for timing (optional)
import math # Added for ceiling function

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Image/Video Classification App",
    page_icon="📹",
    layout="wide"
)

# --- Constants ---
# Define class labels (adjust if your model outputs something different)
CLASS_LABELS = {0: "Non Chatter", 1: "Chatter"}
PREDICTION_THRESHOLD = 0.5 # Threshold for classifying probability

# --- Model Loading ---
@st.cache_resource
def load_classifier_model():
    # IMPORTANT: Use raw string literal or double backslashes for Windows paths
    model_path = r"C:\Users\bhadr\Level0\btp\app\models\vgg19_chatter_multi_input_model.keras"
    # Or: model_path = "C:\\Users\\bhadr\\Level0\\btp\\app\\models\\vgg19_chatter_multi_input_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_classifier_model()

# --- Session State Initialization ---
# To store video processing results and state across reruns
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'video_predictions' not in st.session_state:
    st.session_state.video_predictions = None # Will store list of (timestamp_ms, classification)
if 'video_duration_ms' not in st.session_state:
    st.session_state.video_duration_ms = 0
if 'video_file_bytes' not in st.session_state:
     st.session_state.video_file_bytes = None


# --- Preprocessing Functions ---

# Assume preprocess_uploaded_image exists and works for uploaded image files
# We need a function to preprocess a single frame (numpy array) from the video
# This function should mirror the steps in preprocess_uploaded_image
def preprocess_frame(frame_bgr):
    """
    Preprocesses a single video frame (NumPy array in BGR format).
    Args:
        frame_bgr (np.ndarray): Input frame from cv2.read() (BGR format).
    Returns:
        np.ndarray: Preprocessed frame suitable for model input, or None if error.
    """
    try:
        # 1. Convert BGR to RGB (if your model expects RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize the frame
        frame_resized = cv2.resize(frame_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # 3. Normalize pixel values (adjust if your original preprocessing differs)
        # Example: Scale to [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        # Example: Apply specific normalization if needed (e.g., VGG uses specific means)
        # frame_normalized = tf.keras.applications.vgg19.preprocess_input(frame_resized) # If using VGG preprocess

        # 4. Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)

        # Ensure correct shape (Batch, Height, Width, Channels)
        if frame_batch.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
             return frame_batch.astype(np.float32) # Ensure correct dtype
        else:
             st.warning(f"Preprocessing resulted in unexpected shape: {frame_batch.shape}. Expected: (1, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})")
             return None

    except Exception as e:
        st.error(f"Error preprocessing frame: {e}")
        return None

# --- Sidebar for Numerical Inputs (Common for Image and Video) ---
st.sidebar.header("Numerical Features")
st.sidebar.markdown("Enter the numerical values required for prediction:")

# Define normalization parameters - these should match your training data
# IMPORTANT: Ensure these are the correct values used during training
mean_values = np.array([803.87755102, 446.64489796])
std_values = np.array([533.96406027, 415.35887166])

numerical_features = {
    "feature_1": st.sidebar.number_input("Depth of cut", min_value=0.0, max_value=2000.0, value=5.0, step=0.1, key="num_feat1"),
    "feature_2": st.sidebar.number_input("RPM", min_value=0, max_value=2000, value=50, step=1, key="num_feat2")
}

X_numerical = np.array([[numerical_features["feature_1"], numerical_features["feature_2"]]])
X_normalized = (X_numerical - mean_values) / std_values

st.sidebar.markdown("---")
st.sidebar.caption(f"Raw values: {X_numerical}")
st.sidebar.caption(f"Normalized: {X_normalized}")

def prepare_numerical_features():
    """Returns the normalized numerical features."""
    return X_normalized.astype(np.float32)

st.sidebar.markdown("---")
st.sidebar.header("About the Model")
st.sidebar.markdown("""
This model combines:
- VGG19 feature extraction for images/frames
- Neural network processing for numerical features
- Joint prediction based on both input types
""")

# --- Main App Area ---
st.title("Image & Video Classification (Chatter/Non-Chatter)")

# Mode Selection
processing_mode = st.radio(
    "Select Processing Mode:",
    ("Image Processing", "Video Processing"),
    horizontal=True,
    key="processing_mode"
)

# --- Image Processing Mode ---
if processing_mode == "Image Processing":
    st.header("Image Classification")
    st.markdown("Upload an image and provide numerical values in the sidebar.")
    
    # Reset video state if switching back to image mode
    st.session_state.video_processed = False
    st.session_state.video_predictions = None
    st.session_state.video_duration_ms = 0
    st.session_state.video_file_bytes = None


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Prediction Results")
        if uploaded_file is not None:
            if st.button("Run Image Prediction", key="predict_image"):
                with st.spinner("Processing image..."):
                    try:
                        # Preprocess the image using the original function
                        preprocessed_image = preprocess_uploaded_image(uploaded_file)

                        if preprocessed_image is not None:
                            # Get normalized numerical features
                            numerical_input = prepare_numerical_features()

                            # Make prediction
                            prediction = model.predict([preprocessed_image, numerical_input])
                            probability = float(prediction[0][0])

                            # Classify based on threshold
                            classification_idx = 1 if probability >= PREDICTION_THRESHOLD else 0
                            classification = CLASS_LABELS[classification_idx]

                            # Display results
                            st.success("Prediction Complete!")
                            st.metric("Prediction Probability", f"{probability:.4f}")
                            st.progress(probability)
                            st.subheader(f"Classification: {classification}")
                            if classification == "Chatter":
                                st.warning("Result: Chatter Detected")
                            else:
                                st.success("Result: Non Chatter")
                        else:
                            st.error("Error processing the image. Please try another image.")
                    except Exception as e:
                         st.error(f"An error occurred during prediction: {e}")

            else:
                 # Keep displaying previous results if available, or prompt
                 # This part depends on how you want state handled for images
                 pass

        else:
            st.info("Upload an image and click 'Run Image Prediction'.")


# --- Video Processing Mode ---
elif processing_mode == "Video Processing":
    st.header("Video Classification")
    st.markdown("Upload a video. Each frame will be classified using the numerical values from the sidebar.")

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

    if uploaded_video is not None:
        # Store video bytes in session state to avoid reloading
        if uploaded_video.getvalue() != st.session_state.get('video_file_id', None):
             st.session_state.video_file_bytes = uploaded_video.getvalue()
             st.session_state.video_file_id = uploaded_video.getvalue() # Use content as ID
             st.session_state.video_processed = False # Reset processing state for new video
             st.session_state.video_predictions = None
             st.session_state.video_duration_ms = 0


        process_button_clicked = st.button("Process Video Frames", key="process_video")

        # --- Video Processing Logic ---
        if process_button_clicked and not st.session_state.video_processed:
            with st.spinner("Processing video... This may take a while."):
                try:
                    # Use a temporary file for cv2.VideoCapture
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                        tmpfile.write(st.session_state.video_file_bytes)
                        video_path = tmpfile.name

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error opening video file.")
                    else:
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        st.session_state.video_duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
                        st.info(f"Video details: ~{frame_count} frames, {fps:.2f} FPS, Duration: {st.session_state.video_duration_ms / 1000:.2f}s")

                        predictions = []
                        numerical_input = prepare_numerical_features() # Get numerical features once

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = time.time()

                        processed_count = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break # End of video

                            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                            # Preprocess the frame
                            preprocessed_frame = preprocess_frame(frame) # Use the new function

                            if preprocessed_frame is not None:
                                # Predict using the preprocessed frame and static numerical input
                                prediction = model.predict([preprocessed_frame, numerical_input], verbose=0) # verbose=0 for less console output
                                probability = float(prediction[0][0])
                                classification_idx = 1 if probability >= PREDICTION_THRESHOLD else 0
                                classification = CLASS_LABELS[classification_idx]

                                predictions.append({
                                    "timestamp_ms": timestamp_ms,
                                    "probability": probability,
                                    "classification": classification
                                    })
                            else:
                                st.warning(f"Skipping frame at {timestamp_ms/1000:.2f}s due to preprocessing error.")


                            processed_count += 1
                            progress = min(1.0, processed_count / frame_count) if frame_count > 0 else 0
                            elapsed_time = time.time() - start_time
                            status_text.text(f"Processed frame {processed_count}/{frame_count} ({progress*100:.1f}%) | Time: {elapsed_time:.2f}s")
                            progress_bar.progress(progress)

                        cap.release()
                        os.unlink(video_path) # Clean up temporary file

                        st.session_state.video_predictions = predictions
                        st.session_state.video_processed = True
                        status_text.text(f"Video processing complete! Processed {processed_count} frames in {elapsed_time:.2f}s.")
                        st.success("Video processing finished.")
                        # Force rerun to update UI elements like the slider
                        st.rerun()


                except Exception as e:
                    st.error(f"An error occurred during video processing: {e}")
                    # Clean up temp file if it exists and error occurred
                    if 'video_path' in locals() and os.path.exists(video_path):
                        try:
                             os.unlink(video_path)
                        except Exception as cleanup_e:
                             st.warning(f"Could not delete temp file {video_path}: {cleanup_e}")
                    # Reset state on error
                    st.session_state.video_processed = False
                    st.session_state.video_predictions = None


        # --- Video Playback and Classification Display Area ---
        if st.session_state.video_processed and st.session_state.video_predictions:
            st.subheader("Video Playback & Frame Classification")

            # Display the video using st.video
            # Note: Playback speed control here is informational; user controls it in the browser.
            st.video(st.session_state.video_file_bytes)

            # Informational Playback Speed Selector (Doesn't control st.video)
            playback_speed = st.select_slider(
                 "Suggested Playback Speed (Informational - Use Browser Controls)",
                 options=[0.25, 0.5, 0.75, 1.0],
                 value=1.0,
                 key="playback_speed"
            )
            st.caption("Note: Actual playback speed is controlled by your browser's video player.")


            # Slider for navigating frames/time
            if st.session_state.video_duration_ms > 0:
                 selected_time_ms = st.slider(
                      "Navigate Video Time (milliseconds)",
                      min_value=0,
                      max_value=math.ceil(st.session_state.video_duration_ms), # Use ceiling for max value
                      value=0,
                      step=100, # Adjust step for desired granularity
                      key="video_time_slider"
                 )

                 # Find the prediction closest to the selected time
                 closest_prediction = None
                 min_diff = float('inf')

                 for pred in st.session_state.video_predictions:
                      diff = abs(pred['timestamp_ms'] - selected_time_ms)
                      if diff < min_diff:
                           min_diff = diff
                           closest_prediction = pred

                 # Display the classification for the selected time
                 if closest_prediction:
                      st.markdown("---")
                      st.markdown(f"**Classification at ~{closest_prediction['timestamp_ms']/1000:.2f} seconds:**")
                      classification = closest_prediction['classification']
                      probability = closest_prediction['probability']

                      st.metric("Frame Classification", classification)
                      st.metric("Frame Probability", f"{probability:.4f}")
                      if classification == "Chatter":
                            st.warning("Result: Chatter")
                      else:
                            st.success("Result: Non Chatter")
                      # Optional: Show small preview of the frame if needed (more complex)

                 else:
                      st.info("Slide the timeline to view frame classifications.")

            else:
                 st.warning("Video duration calculation failed, cannot display timeline slider.")

        elif st.session_state.video_file_bytes is not None and not st.session_state.video_processed:
             st.info("Click 'Process Video Frames' to analyze the video.")


    else:
        st.info("Upload a video file to begin.")
        # Clear previous video state if no video is uploaded
        st.session_state.video_processed = False
        st.session_state.video_predictions = None
        st.session_state.video_duration_ms = 0
        st.session_state.video_file_bytes = None


# --- Footer or additional info ---
st.markdown("---")
st.caption("Developed for BTP | Uses TensorFlow/Keras & Streamlit | Handles Image & Video")