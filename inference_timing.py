import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import time
import os
# import traceback # Uncomment for detailed error tracebacks

# --- Configuration ---

# !! REQUIRED: Update this path to your actual model file !!
MODEL_PATH = r"C:\Users\bhadr\Level0\btp\app\models\vgg19_chatter_multi_input_model.keras" # Example path

# !! REQUIRED: Update this path to a sample image file !!
SAMPLE_IMAGE_PATH = r"C:\Users\bhadr\Level0\btp\infimage.jpg" # <--- CHANGE THIS

# Normalization parameters (copied from your Streamlit app)
MEAN_VALUES = np.array([803.87755102, 446.64489796])
STD_VALUES = np.array([533.96406027, 415.35887166])

# Example numerical inputs (RAW values before normalization)
EXAMPLE_NUMERICAL_RAW = np.array([[5.0, 50.0]]) # e.g., Depth=5.0, RPM=50

# --- Measurement Parameters ---
NUM_ITERATIONS = 100  # How many times to run for averaging
NUM_WARMUP = 10     # How many initial runs to discard (warming up preprocessing + inference)

# --- Preprocessing Functions ---
# Ensure these accurately reflect your Streamlit app's preprocessing logic

def preprocess_image_for_timing(image_path, target_height, target_width, target_channels):
    """
    Loads, decodes, resizes, normalizes (basic), and batches an image.
    *** ADAPT THIS TO MATCH YOUR ACTUAL PREPROCESSING ***
    Especially the normalization step (e.g., / 255.0 or VGG-specific).
    Returns None on error.
    """
    try:
        img = tf.io.read_file(image_path) # Reading file is part of the process
        img = tf.image.decode_image(img, channels=target_channels, expand_animations=False)
        img = tf.image.resize(img, [target_height, target_width])
        # == Crucial Step: Match your training preprocessing ==
        img = tf.keras.applications.vgg19.preprocess_input(tf.cast(img, tf.float32)) # Example: VGG19
        img = tf.expand_dims(img, axis=0)
        return img
    except Exception as e:
        # Print error only once to avoid flooding console during loop
        # print(f"Error preprocessing image '{image_path}': {e}")
        return None

def preprocess_numerical(raw_numerical_data, means, stds):
     """Applies normalization based on Streamlit app logic."""
     normalized = (raw_numerical_data - means) / stds
     return normalized.astype(np.float32)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Total Processing Time Measurement (Preprocessing + Inference) ---")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 1. Check Files and Paths ---
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
        exit()
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"FATAL ERROR: Sample image not found at '{SAMPLE_IMAGE_PATH}'")
        exit()
    print(f"Using model: {MODEL_PATH}")
    print(f"Using sample image: {SAMPLE_IMAGE_PATH}")

    # --- 2. Confirm CPU Execution ---
    print("Available devices:", tf.config.list_physical_devices())
    print("Ensuring execution on CPU.")

    # --- 3. Load Model ---
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        exit()

    # --- 4. Determine Input Shapes from Model ---
    print("\nInspecting model input shapes...")
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = None, None, None # Define before try block
    try:
        if not isinstance(model.input_shape, list):
             raise TypeError(f"Expected model.input_shape to be a list, but got {type(model.input_shape)}. Shape: {model.input_shape}")
        input_shapes = model.input_shape
        image_input_shape, numerical_input_shape = None, None
        print(f"Model expects {len(input_shapes)} inputs with shapes: {input_shapes}")
        for shape in input_shapes:
            if len(shape) == 4 and None not in shape[1:3]: image_input_shape = shape
            elif len(shape) == 2: numerical_input_shape = shape
        if image_input_shape is None: raise ValueError("Could not determine image input shape.")
        if numerical_input_shape is None: raise ValueError("Could not determine numerical input shape.")

        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = image_input_shape[1:4]
        print(f"Determined Image Input Shape: {image_input_shape} -> H={IMAGE_HEIGHT}, W={IMAGE_WIDTH}, C={IMAGE_CHANNELS}")
        print(f"Determined Numerical Input Shape: {numerical_input_shape}")
        expected_num_features = numerical_input_shape[1]
        provided_num_features = EXAMPLE_NUMERICAL_RAW.shape[1]
        if expected_num_features is not None and expected_num_features != provided_num_features:
             print(f"FATAL ERROR: Model expects {expected_num_features} numerical features, EXAMPLE_NUMERICAL_RAW has {provided_num_features}.")
             exit()

    except Exception as e:
        print(f"\nFATAL ERROR inspecting model input shapes: {e}")
        exit()

    # --- 5. Prepare Static Numerical Input (can be done once outside loop) ---
    # Preprocessing numerical features is usually very fast, but we do it once here
    # If numerical inputs were changing per frame, this would move inside the loop
    try:
        numerical_input = preprocess_numerical(EXAMPLE_NUMERICAL_RAW, MEAN_VALUES, STD_VALUES)
        print(f"\nNumerical features preprocessed once. Shape: {numerical_input.shape}, Dtype: {numerical_input.dtype}")
    except Exception as e:
        print(f"FATAL ERROR preparing numerical input data: {e}")
        exit()

    # --- 6. Warm-up Phase (includes preprocessing) ---
    print(f"\nRunning {NUM_WARMUP} warm-up iterations (Preprocessing + Inference)...")
    warmup_error = False
    try:
        for i in range(NUM_WARMUP):
            # Preprocess image inside warm-up
            preprocessed_image = preprocess_image_for_timing(SAMPLE_IMAGE_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
            if preprocessed_image is None:
                 warmup_error = True
                 print(f"\nError during image preprocessing in warm-up iteration {i+1}. Check image path and format.")
                 break # Stop warm-up if preprocessing fails

            # Inference
            _ = model.predict([preprocessed_image, numerical_input], verbose=0)
            print(f" Warm-up {i+1}/{NUM_WARMUP} done.", end='\r')

        if not warmup_error:
            print("\nWarm-up complete. ")
        else:
             print("\nWarm-up failed due to preprocessing error.")
             exit()

    except Exception as e:
        print(f"\nFATAL ERROR during warm-up phase: {e}")
        # traceback.print_exc() # Uncomment for detailed traceback
        exit()

    # --- 7. Timing Loop (Preprocessing + Inference + Optional Post-processing) ---
    print(f"Running {NUM_ITERATIONS} timed iterations...")
    processing_times = []
    preprocessing_error_count = 0
    inference_error_count = 0

    for i in range(NUM_ITERATIONS):
        try:
            start_time = time.perf_counter()

            # Step 1 & 2a: Load and preprocess image
            preprocessed_image = preprocess_image_for_timing(SAMPLE_IMAGE_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
            if preprocessed_image is None:
                 preprocessing_error_count += 1
                 if preprocessing_error_count == 1: # Print only the first time
                      print("\nError during image preprocessing in timed loop. Skipping iteration.")
                 continue # Skip this iteration

            # Step 2b: Numerical features are already preprocessed (numerical_input)

            # Step 3: Model Inference
            try:
                prediction = model.predict([preprocessed_image, numerical_input], verbose=0)
            except Exception as ie:
                 inference_error_count += 1
                 if inference_error_count == 1:
                      print(f"\nError during model.predict() in timed loop: {ie}. Skipping iteration.")
                 continue # Skip this iteration


            # Step 4 (Optional): Basic post-processing example
            # probability = float(prediction[0][0])
            # classification = "Chatter" if probability >= 0.5 else "Non Chatter"
            # Add any other simple steps you do after getting the prediction

            end_time = time.perf_counter()
            duration = end_time - start_time
            processing_times.append(duration)

            print(f" Iteration {i + 1}/{NUM_ITERATIONS}: {duration * 1000:.2f} ms", end='\r')

        except Exception as e:
            # Catch unexpected errors during the loop timing itself
            print(f"\nUnexpected error in timing loop at iteration {i + 1}: {e}")
            # traceback.print_exc()
            break # Stop timing loop on unexpected errors

    print("\nTimed iterations complete. ")

    # Print error summaries if any occurred
    if preprocessing_error_count > 0:
         print(f"Warning: Skipped {preprocessing_error_count} iterations due to image preprocessing errors.")
    if inference_error_count > 0:
         print(f"Warning: Skipped {inference_error_count} iterations due to model inference errors.")

    # --- 8. Calculate and Display Results ---
    print("\n--- Total Processing Time Results (Preprocessing + Inference) ---")
    if not processing_times:
        print("No successful iterations were timed.")
    else:
        processing_times_np = np.array(processing_times)

        average_time_sec = np.mean(processing_times_np)
        std_dev_time_sec = np.std(processing_times_np)
        min_time_sec = np.min(processing_times_np)
        max_time_sec = np.max(processing_times_np)
        median_time_sec = np.median(processing_times_np)

        avg_ms = average_time_sec * 1000
        std_ms = std_dev_time_sec * 1000
        min_ms = min_time_sec * 1000
        max_ms = max_time_sec * 1000
        med_ms = median_time_sec * 1000

        print(f"Number of successfully timed iterations: {len(processing_times_np)}")
        print(f"Average total processing time: {avg_ms:.2f} ms")
        print(f"Median total processing time:  {med_ms:.2f} ms")
        print(f"Standard deviation:            {std_ms:.2f} ms")
        print(f"Minimum total processing time: {min_ms:.2f} ms")
        print(f"Maximum total processing time: {max_ms:.2f} ms")

        if average_time_sec > 0:
            max_fps_total = 1.0 / average_time_sec
            print(f"\nTheoretical Max FPS (based ONLY on average total processing time): {max_fps_total:.2f} FPS")
            print(f"This implies the full pipeline (load+preprocess+predict) requires ~{avg_ms:.2f} ms per frame.")
        else:
            print("\nCould not calculate max FPS (average time was zero or negative).")

    print("\n--- Important Notes ---")
    print("* This measurement includes image reading, preprocessing, and model.predict().")
    print("* It does NOT include time spent by an application framework (like Streamlit) for handling uploads, UI updates, etc.")
    print("* Actual real-time throughput may still be lower due to camera capture latency (if applicable) and system overhead.")
    print("* Ensure the image preprocessing logic in `preprocess_image_for_timing` EXACTLY matches your application.")