# Camera Face Detection and Recognition App

This is a simple **real-time face detection and recognition** application built using Python, OpenCV, and Tkinter.

## Features
- **Real-time Face Detection**: Utilizes OpenCV's Haar Cascade classifier to detect faces from the live camera feed.
- **Face Recognition**: Compares detected faces with pre-encoded faces from the `images` folder and identifies known individuals.
- **Tkinter GUI**: Provides a graphical user interface to display the live video feed.
- **Add New Faces**: Add new face images to the `images` folder, and the application will automatically process them for recognition.

## Dependencies
- `opencv-python`
- `numpy`
- `Pillow`
- `screeninfo`
- `tkinter` (comes pre-installed with Python)

## Installation and Running the App
1. Clone this repository:
   ```bash
   git clone https://github.com/IvanYanishevskyi/FaceDetectionAPP.git
   cd FaceDetectionAPP
   ```
2. ```bash
   python main.py
   ```
### How It Works

The app captures video from the default camera.
It detects faces in real-time using OpenCV's Haar Cascade classifier.
Detected faces are compared with known encodings stored as flattened grayscale images from the images folder.
Recognized faces are labeled with their corresponding names on the video feed.
