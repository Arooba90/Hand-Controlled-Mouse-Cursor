import cv2
import mediapipe as mp
import pyautogui
import math
# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # Use the webcam

# Get the screen size
screen_width, screen_height = pyautogui.size()

# Setup MediaPipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a later mirror view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        result = hands.process(rgb_frame)

        # If hands are detected, extract the landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the position of the index finger tip (landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                 # Get the position of the thumb tip (landmark 4)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Calculate the Euclidean distance between the index and thumb tips
                distance = math.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)


                # Map the normalized coordinates to the screen size
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)

                # Move the mouse to the new coordinates
                pyautogui.moveTo(x, y)

                # Check if the distance between index and thumb is below a threshold to simulate a click
                click_threshold = 0.04
                if distance < click_threshold:
                    pyautogui.click()  # Trigger a click

        # Display the frame with landmarks
        cv2.imshow("Hand Tracking - Mouse Control", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
