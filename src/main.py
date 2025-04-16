import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detect up to 2 hands
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Function to calculate the distance between two points
def calculate_distance(a, b):
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

# Function to recognize gestures based on distances
def recognize_gesture(hand_landmarks):
    # Get the coordinates of key landmarks for gesture recognition
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculate distances between key landmarks
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    index_middle_dist = calculate_distance(index_tip, middle_tip)
    middle_ring_dist = calculate_distance(middle_tip, ring_tip)
    ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
    
    # Recognizing Thumbs Up
    if thumb_index_dist > 0.1 and index_middle_dist > 0.1 and ring_pinky_dist < 0.05:
        return "Thumbs Up"
    
    # Recognizing Fist
    if thumb_index_dist < 0.05 and index_middle_dist < 0.05 and ring_pinky_dist < 0.05:
        return "Fist"
    
    # Recognizing Open Hand (All fingers extended)
    if thumb_index_dist > 0.1 and index_middle_dist > 0.1 and middle_ring_dist > 0.1 and ring_pinky_dist > 0.1:
        return "Open Hand"
    
    # Recognizing Peace Sign (Index and middle fingers extended)
    if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and \
         ring_tip.y > index_tip.y and pinky_tip.y > middle_tip.y:
        return "Peace Sign"
    
    # Recognizing OK Sign (Thumb and index finger touching)
    if thumb_index_dist < 0.05:
        return "OK Sign"
    
    # Recognizing Rock On (Pinky and Index fingers extended)
    if index_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y and \
         abs(index_tip.x - pinky_tip.x) < 0.05:
        return "Rock On"
    
    # Default case if no gesture matches
    return "None"

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

            # Recognize gesture
            gesture = recognize_gesture(hand_landmarks)

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
