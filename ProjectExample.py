import cv2  # Import OpenCV library for image processing
import mediapipe as mp  # Import MediaPipe library for hand tracking
import time  # Import time library to measure FPS
import HandTrackingModule as htm  # Import custom hand tracking module

# Initialize variables to calculate FPS
pTime = 0  # Previous time
cTime = 0  # Current time

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)  # Try different indices: 0, 1, 2, etc.

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Create an object of the handDetector class from HandTrackingModule
detector = htm.handDetector()

# Main loop to process each frame
while True:
    success, img = cap.read()  # Capture a frame from the camera
    if not success or img is None:
        print("Failed to capture image")
        continue

    # Detect hands and draw landmarks on the image
    img = detector.findHands(img, draw=True)
    
    # Get the positions of landmarks
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])  # Print the position of the fifth landmark (index 4)

    # Calculate the frames per second (FPS)
    cTime = time.time()  # Get the current time
    fps = 1 / (cTime - pTime)  # Calculate FPS
    pTime = cTime  # Update the previous time

    # Display the FPS on the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for 1 ms and check if 'q' is pressed
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
