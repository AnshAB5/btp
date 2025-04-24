import cv2
import os
from tkinter import Tk, filedialog
from PIL import Image

def create_video_from_images(image_folder, output_folder, fps=5):
    """
    Creates a video from all images in a given folder.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where the video will be saved.
        fps (int): Frames per second for the output video. Defaults to 5.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images.sort()  # Ensure images are processed in alphabetical order

    if not images:
        print(f"No image files found in the folder: {image_folder}")
        return

    try:
        # Open the first image to get its dimensions
        first_image_path = os.path.join(image_folder, images[0])
        img = Image.open(first_image_path)
        width, height = img.size

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_filename = os.path.join(output_folder, 'output_video.mp4')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        for image_name in images:
            image_path = os.path.join(image_folder, image_name)
            frame = cv2.imread(image_path)
            if frame is not None:
                out.write(frame)
            else:
                print(f"Error reading image: {image_path}")

        # Release the VideoWriter object
        out.release()
        print(f"Video '{video_filename}' created successfully in '{output_folder}' with {fps} FPS.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()  # Hide the main tkinter window

    image_folder_selected = filedialog.askdirectory(title="Select the folder containing images")
    if not image_folder_selected:
        print("Image folder not selected. Exiting.")
        exit()

    output_folder_selected = filedialog.askdirectory(title="Select the folder to save the video")
    if not output_folder_selected:
        print("Output folder not selected. Exiting.")
        exit()

    create_video_from_images(image_folder_selected, output_folder_selected, fps=5)