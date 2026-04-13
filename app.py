import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="PostureGhost", page_icon="👻", layout="wide")

st.title("👻 PostureGhost")
st.markdown("### Votre coach posture IA qui vous montre le chemin")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📹 Webcam en direct")
    run = st.button("🚀 Démarrer la détection", type="primary")
    stop = st.button("⏹️ Arrêter")

with col2:
    st.markdown("### 📊 Votre score posture")
    score_placeholder = st.empty()
    alert_placeholder = st.empty()

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def draw_ghost(frame, landmarks, w, h):
    ghost_offset_x = 200
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = (int(landmarks[start_idx].x * w + ghost_offset_x), int(landmarks[start_idx].y * h))
            end = (int(landmarks[end_idx].x * w + ghost_offset_x), int(landmarks[end_idx].y * h))
            cv2.line(frame, start, end, (200, 200, 255), 3)
    return frame

if run:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    
    while not stop:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Calcul score posture
            left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
            left_hip = [landmarks[23].x * w, landmarks[23].y * h]
            back_angle = calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] - 50])
            posture_score = max(0, 100 - abs(90 - back_angle) * 2)
            
            # Mettre à jour l'interface
            score_placeholder.metric("Score posture", f"{posture_score}%")
            if posture_score < 70:
                alert_placeholder.warning("⚠️ Redressez-vous ! Suivez le fantôme 👻")
            else:
                alert_placeholder.success("✅ Posture parfaite !")
            
            # Dessiner fantôme et squelette
            frame = draw_ghost(frame, landmarks, w, h)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
    
    cap.release()