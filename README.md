# driver_drowsiness_checker
# Driver Drowsiness Detection

This project uses computer vision and machine learning techniques to detect signs of drowsiness in drivers. It can be used to alert drivers when they show signs of fatigue, promoting safety on the road.

## Features
- Detects the driver's face using a webcam or video input.
- Analyzes eye movement to determine if the driver is drowsy.
- Provides real-time feedback and visual alerts if drowsiness is detected.

## Requirements
Make sure you have the following installed:

- Python 3.x
- OpenCV
- TensorFlow (if using deep learning models)
- dlib (for face and facial landmark detection)
- numpy

Install the required libraries using:


pip install opencv-python dlib tensorflow numpy

## How It Works
The driver_drowsiness.py script performs the following steps:

Face Detection: Uses OpenCV or dlib to detect the face of the driver in real-time.

Eye Aspect Ratio (EAR): Calculates EAR from facial landmarks to determine if the driver's eyes are closed for too long.

Alert System: If EAR falls below a threshold for a sustained time, a visual or audio alert is triggered.

Contributing
Feel free to fork the repository, create issues, and submit pull requests. Contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

