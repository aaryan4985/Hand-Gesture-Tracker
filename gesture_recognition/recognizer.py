# hand-gesture-tracker/gesture_recognition/recognizer.py

import math
from utils.math_utils import calculate_distance

class GestureRecognizer:
    def __init__(self, mp_hands):
        self.mp_hands = mp_hands

    def recognize(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        thumb_index_dist = calculate_distance(thumb_tip, index_tip)
        index_middle_dist = calculate_distance(index_tip, middle_tip)
        middle_ring_dist = calculate_distance(middle_tip, ring_tip)
        ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)

        if thumb_index_dist > 0.1 and index_middle_dist > 0.1 and ring_pinky_dist < 0.05:
            return "Thumbs Up"
        if thumb_index_dist < 0.05 and index_middle_dist < 0.05 and ring_pinky_dist < 0.05:
            return "Fist"
        if thumb_index_dist > 0.1 and index_middle_dist > 0.1 and middle_ring_dist > 0.1 and ring_pinky_dist > 0.1:
            return "Open Hand"
        if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and \
           ring_tip.y > index_tip.y and pinky_tip.y > middle_tip.y:
            return "Peace Sign"
        if thumb_index_dist < 0.05:
            return "OK Sign"
        if index_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y and \
           abs(index_tip.x - pinky_tip.x) < 0.05:
            return "Rock On"

        return "None"
