import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def extract_hand_landmarks_from_image(image, max_num_hands=1):
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=max_num_hands,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

def preprocess_landmarks(landmarks):
    if landmarks is None:
        return None
    lm = landmarks.reshape(-1, 3).copy()
    wrist = lm[0].copy()
    lm[:, :2] -= wrist[:2]
    maxv = np.max(np.abs(lm[:, :2]))
    if maxv > 0:
        lm[:, :2] = lm[:, :2] / maxv
    return lm.flatten()
