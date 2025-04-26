import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get screen resolution (width and height)
screen_width, screen_height = pyautogui.size()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand model
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index and thumb tips landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the distance between the thumb and index tip
            distance = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)

            # Map hand landmarks to screen size
            # Convert normalized coordinates to pixel values based on screen size
            index_x = int(index_tip.x * screen_width)
            index_y = int(index_tip.y * screen_height)

            # Move the mouse to the new position on the screen
            pyautogui.moveTo(index_x, index_y)

            # If the distance between thumb and index finger is below a threshold, click
            if distance < 0.05:
                pyautogui.click()

    # Display the webcam feed with hand landmarks
    cv2.imshow("Virtual Mouse", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
