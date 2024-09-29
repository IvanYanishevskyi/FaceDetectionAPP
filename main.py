import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from screeninfo import get_monitors
import os

# Class definition for the camera application
class CameraApp:
    def __init__(self, window):
        # Get the monitor information and set the main window title
        monitors = get_monitors()
        my_monitor = monitors[0]
        self.window = window
        self.window.title("Camera App")

        # Initialize video capture and load the Haar Cascade for face detection
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        # Create a canvas to display the video feed
        self.canvas = tk.Canvas(window, width=my_monitor.width, height=my_monitor.height)
        self.canvas.pack()

        # Load known face encodings from the images folder
        self.known_face_encodings = self.load_known_faces("images")

        # Start updating the video feed
        self.update()
        self.window.mainloop()

    # Load known face encodings from a folder of images
    def load_known_faces(self, folder):
        encodings = []
        for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = cv2.imread(os.path.join(folder, filename))
                encoding = self.get_face_encoding(image)
                if encoding is not None:
                    encodings.append((filename.split('.')[0], encoding))  # Store name and encoding
        return encodings

    # Get the face encoding by detecting faces and extracting the facial region
    def get_face_encoding(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)  # Detect faces
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Take the first detected face
            face = gray_image[y:y + h, x:x + w]  # Extract the face region
            return cv2.resize(face, (100, 100)).flatten()  # Resize and flatten for encoding
        return None

    # Update the video feed and process face detection and recognition
    def update(self):
        ret, frame = self.vid.read()  # Read the video frame
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            detected_faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10)  # Detect faces

            # For each detected face, perform recognition by comparing encodings
            for (x, y, w, h) in detected_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face
                detected_face = gray_frame[y:y + h, x:x + w]  # Extract the face region
                detected_face_encoding = cv2.resize(detected_face, (100, 100)).flatten()  # Get encoding of the detected face

                # Compare the detected face with known faces
                for name, known_encoding in self.known_face_encodings:
                    distance = np.linalg.norm(detected_face_encoding - known_encoding)  # Calculate distance between encodings
                    if distance < 20000:  # Threshold for face recognition
                        cv2.putText(frame, name, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Label the face

            # Convert the frame to an image and display it on the canvas
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Schedule the next update after 10 ms
        self.window.after(10, self.update)

    # Release the video capture object when the application is closed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Initialize the Tkinter window and start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
