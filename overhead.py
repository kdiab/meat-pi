import cv2
import numpy as np
import random

# Load the pre-trained Haar cascade for frontal face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open the video capture (0 for default camera)
cap = cv2.VideoCapture(0)
# Set a lower capture resolution for performance improvement
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# List to store tracked faces with their centroids and assigned stable colors
tracked_faces = []  # Each element: {'centroid': (cx, cy), 'color': (B, G, R)}
DISTANCE_THRESHOLD = 30  # Threshold in pixels for matching faces between frames

# Create a resizable window and set it to full-screen mode
cv2.namedWindow("Head Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Head Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR so that colored rectangles can be drawn
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Detect faces (heads) using the Haar cascade on the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    new_tracked_faces = []
    for (x, y, w, h) in faces:
        # Compute the centroid of the detected face
        cx, cy = (x + w / 2, y + h / 2)
        assigned_color = None

        # Try to match this face with one from the previous frame to reuse its color
        for tf in tracked_faces:
            if np.linalg.norm(np.array((cx, cy)) - np.array(tf['centroid'])) < DISTANCE_THRESHOLD:
                assigned_color = tf['color']
                break

        # If no match is found, assign a new random color
        if assigned_color is None:
            assigned_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

        # Update the tracker for this frame
        new_tracked_faces.append({'centroid': (cx, cy), 'color': assigned_color})

        # Blur the region inside the detected face in the grayscale (BGR) image
        roi = gray_bgr[y:y+h, x:x+w]
        roi_blurred = cv2.GaussianBlur(roi, (25, 25), 0)
        gray_bgr[y:y+h, x:x+w] = roi_blurred

        # Draw a rectangle around the face with the stable random color
        cv2.rectangle(gray_bgr, (x, y), (x+w, y+h), assigned_color, 2)

    # Update the tracked faces for the next frame
    tracked_faces = new_tracked_faces

    # Resize the processed frame to 1920x1080 for display
    resized_frame = cv2.resize(gray_bgr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Head Detection", resized_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()

