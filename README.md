# Computer_vision_pose_estimation

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
from google.colab.patches import cv2_imshow

# Specify the direct download link of the image file from Google Drive
image_url = 'https://drive.google.com/uc?id=1VfGwdE0D8j-Suyheced02d3U8VwZPiMF'

# Read the image from the URL
req = urllib.request.urlopen(image_url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1)

# Check if the image was loaded successfully
if image is None:
    print("Failed to load image from URL.")
else:
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the Mediapipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Run pose estimation
    results = pose.process(image_rgb)

    # Get the keypoints
    keypoints = results.pose_landmarks

    # Draw the keypoints on the image
    if keypoints is not None:
        for landmark in keypoints.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

    # Display the resulting image
    cv2_imshow(image)
