import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detect up to 2 hands
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle)

while True:
    success, frame = cap.read()  # Read a frame from the webcam
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB (because MediaPipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of key landmarks for gesture recognition
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate angles to recognize gestures
            # Example: Recognizing a "thumbs up" gesture based on the thumb
            thumb_angle = calculate_angle(
                (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y),
                (thumb_tip.x, thumb_tip.y),
                (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y)
            )

            # Gesture Detection Logic
            if thumb_angle < 30:
                gesture = "Thumbs Up"
            elif all([index_tip.y < thumb_tip.y, middle_tip.y < thumb_tip.y, ring_tip.y < thumb_tip.y, pinky_tip.y < thumb_tip.y]):
                gesture = "Fist"
            else:
                gesture = "None"

            # Display the recognized gesture on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with landmarks and gesture text
    cv2.imshow("Hand Gesture Tracker", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
