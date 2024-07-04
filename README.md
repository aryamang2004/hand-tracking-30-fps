# Hand Tracking at 30 FPS with OpenCV and MediaPipe

This project demonstrates real-time hand tracking at 30 FPS using OpenCV and MediaPipe. The code captures video from a webcam, detects hand landmarks, and displays them along with the frames per second (FPS).

## Prerequisites

Before running the program, ensure you have the following installed:

- **Python 3.x**: Ensure you have the latest version of Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **OpenCV**: A powerful library for image processing.
- **MediaPipe**: A versatile library for building multimodal, cross-platform machine learning pipelines.

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe
```

## Project Structure

The project consists of three main Python files:

1. **Basics.py**: Demonstrates basic hand tracking using MediaPipe directly.
2. **HandTrackingModule.py**: Contains the hand tracking logic using MediaPipe.
3. **ProjectExample.py**: Uses the hand tracking module to display hand landmarks and FPS.

### Basics.py

This file demonstrates how to use MediaPipe directly for hand tracking without using the custom module. Here’s what it does:

- **Initialization**:
  - Initializes the MediaPipe Hands solution.
  - Sets up video capture from the webcam.
  
  - **MediaPipe Hands**: Initializes the hands model.
  - **Video Capture**: Sets up the webcam for real-time video capture.

- **Main Loop**:
  - Captures frames from the webcam.
  - Converts frames to RGB format.
  - Processes frames to detect hand landmarks.
  - Draws landmarks on the frames.
  - Calculates and displays FPS.
  - Shows the video frames with hand landmarks and FPS.
  
  - **Hand Landmark Detection**: Detects landmarks in each frame and overlays them on the video feed.
  - **FPS Display**: Displays the current FPS on the video feed for performance monitoring.


### HandTrackingModule.py

This file defines a class called `handDetector` which encapsulates the hand tracking functionality using MediaPipe. Here’s a detailed explanation of the components:

- **Initialization (`__init__` method)**:
  - Initializes the MediaPipe Hands solution with parameters such as mode, maximum number of hands to detect, detection confidence, and tracking confidence.
  - Sets up the drawing utilities to visualize hand landmarks.
  
  - **`mode`**: A boolean to specify if the detection should run in static image mode or not. If false, it runs in video mode.
  - **`maxHands`**: The maximum number of hands to detect.
  - **`detectionCon`**: Minimum detection confidence threshold.
  - **`trackCon`**: Minimum tracking confidence threshold.

- **Finding Hands (`findHands` method)**:
  - Converts the input image from BGR to RGB format.
  - Processes the RGB image to detect hand landmarks.
  - Optionally draws the detected hand landmarks on the input image.
  
  - **Converting to RGB**: MediaPipe requires images in RGB format.
  - **Drawing Landmarks**: Uses MediaPipe's drawing utilities to overlay the detected landmarks on the original image.

- **Finding Position (`findPosition` method)**:
  - Extracts the pixel coordinates of each hand landmark detected in the image.
  - Optionally draws small circles at each landmark position.
  - Returns a list of landmark positions.
  
  - **Landmark Coordinates**: Converts normalized landmark coordinates to pixel values.

- **Main Function**:
  - Captures video frames from the webcam.
  - Uses the `handDetector` to detect hands in each frame.
  - Displays the hand landmarks and calculates the FPS.
  - Shows the processed video frames with landmarks and FPS on the screen.
  
  - **FPS Calculation**: Uses the time library to calculate the number of frames processed per second.

### ProjectExample.py

This file showcases the usage of the `handDetector` class from `HandTrackingModule.py`. Here’s what it covers:

- **Initialization**:
  - Sets up video capture from the webcam.
  - Creates an instance of the `handDetector` class.
  
  - **Video Capture**: Initializes the webcam.
  - **handDetector Instance**: Uses the custom hand detection class.

- **Main Loop**:
  - Captures frames from the webcam.
  - Uses the `handDetector` to detect hands in each frame.
  - Extracts the positions of hand landmarks.
  - Prints the position of a specific landmark.
  - Calculates and displays FPS.
  - Shows the processed video frames with landmarks and FPS.
  
  - **Hand Detection and Tracking**: Uses the custom module to process each frame.
  - **Landmark Position Extraction**: Extracts and prints the position of specific landmarks.
  - **FPS Calculation and Display**: Monitors and displays the system’s performance in real-time.

## Running the Project

1. **Ensure all three files (`Basics.py`, `HandTrackingModule.py`, and `ProjectExample.py`) are in the same directory.**

2. **Open a terminal or command prompt in the project directory.**

3. **Run the `ProjectExample.py` script to start the hand tracking program:**

   ```bash
   python ProjectExample.py
   ```

4. **The program will start capturing video from your webcam and display the hand landmarks along with the FPS.**

5. **Press 'q' to exit the program.**

## Detailed Steps to Run the Program

### 1. Setting Up the Environment

Make sure you have Python 3.x installed. If not, download and install it from [python.org](https://www.python.org/downloads/).

Install the required packages using pip:

```bash
pip install opencv-python mediapipe
```

### 2. Creating the Necessary Files

Create the following Python files in your project directory:

- **Basics.py**: Demonstrates basic hand tracking using MediaPipe.
- **HandTrackingModule.py**: Contains the class for hand detection.
- **ProjectExample.py**: Uses the `handDetector` class for hand tracking and displays the results.

### 3. Writing the Code

- **Basics.py**:
  - Initialize MediaPipe Hands.
  - Set up video capture from the webcam.
  - Capture frames, process them for hand landmarks, and display the results with FPS.

- **HandTrackingModule.py**:
  - Define a class called `handDetector`.
  - Initialize MediaPipe Hands and drawing utilities.
  - Implement methods to find hands and extract landmark positions.

- **ProjectExample.py**:
  - Set up video capture.
  - Create an instance of the `handDetector` class.
  - Capture frames, use the `handDetector` to process them, and display the results with FPS.

### 4. Running the Program

Open a terminal or command prompt in the project directory and run the `ProjectExample.py` script:

```bash
python ProjectExample.py
```

The program will start capturing video from your webcam and display the hand landmarks along with the FPS. Press 'q' to exit the program.

## Conclusion

This project provides a simple yet powerful demonstration of real-time hand tracking using OpenCV and MediaPipe. By following the steps outlined in this guide, you can easily set up and run the hand tracking program on your own system. Feel free to modify and expand the code to suit your needs.
