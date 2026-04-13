"""
POSTUREGHOST - Coach posture IA avec fantôme
Challenge: Wellness & Agent Challenge GIEW 2026
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

FRAME_SKIP = 1
RESOLUTION_WIDTH  = 1280
RESOLUTION_HEIGHT = 720

GOOD_POSTURE_THRESHOLD = 75
WARNING_THRESHOLD      = 55
CRITICAL_THRESHOLD     = 35

COLOR_GREEN      = (0, 220, 100)
COLOR_RED        = (50, 50, 230)
COLOR_ORANGE     = (0, 140, 255)
COLOR_WHITE      = (255, 255, 255)
COLOR_CYAN       = (220, 220, 0)
COLOR_GHOST      = (255, 200, 180)
COLOR_GHOST_BONE = (200, 170, 255)
COLOR_PANEL_BG   = (30, 30, 45)

# ============================================
# MEDIAPIPE
# ============================================

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ============================================
# CALCUL ANGLES & SCORE
# ============================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def calculate_posture_score(landmarks, frame_shape):
    h, w, _ = frame_shape

    def pt(idx):
        lm = landmarks[idx]
        return [lm.x * w, lm.y * h]

    ls   = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs   = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    lh   = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    rh   = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
    le   = pt(mp_pose.PoseLandmark.LEFT_EAR.value)
    re   = pt(mp_pose.PoseLandmark.RIGHT_EAR.value)

    # Angle dos (épaule → hanche → vertical bas)
    back_angle = (
        calculate_angle(ls, lh, [lh[0], lh[1] - 100]) +
        calculate_angle(rs, rh, [rh[0], rh[1] - 100])
    ) / 2

    # Avancement tête
    ear_mid = [(le[0]+re[0])/2, (le[1]+re[1])/2]
    sh_mid  = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
    head_fwd = ear_mid[0] - sh_mid[0]

    # Asymétrie épaules
    shoulder_tilt = abs(ls[1] - rs[1])

    back_score     = max(0, 100 - back_angle * 1.8)
    head_score     = max(0, 100 - abs(head_fwd) * 0.9)
    shoulder_score = max(0, 100 - shoulder_tilt * 2.0)

    score = int(back_score * 0.50 + head_score * 0.30 + shoulder_score * 0.20)
    return max(0, min(100, score)), back_angle, head_fwd, shoulder_tilt


def get_status(score):
    if score >= GOOD_POSTURE_THRESHOLD:
        return "PARFAIT",   COLOR_GREEN
    elif score >= WARNING_THRESHOLD:
        return "ATTENTION", COLOR_ORANGE
    else:
        return "DANGER",    COLOR_RED

# ============================================
# HELPERS DESSIN
# ============================================

def rounded_rect(img, x1, y1, x2, y2, color, r=12, fill=True, alpha=1.0):
    if alpha < 1.0:
        ov = img.copy()
        rounded_rect(ov, x1, y1, x2, y2, color, r, fill, 1.0)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
        return
    t = -1 if fill else 2
    if fill:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1+r,y1),(x2-r,y1), color, 2)
        cv2.rectangle(img, (x1+r,y2),(x2-r,y2), color, 2)
        cv2.rectangle(img, (x1,y1+r),(x1,y2-r), color, 2)
        cv2.rectangle(img, (x2,y1+r),(x2,y2-r), color, 2)
        cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,2)
        cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,2)
        cv2.ellipse(img,(x1+r,y2-r),(r,r), 90,0,90,color,2)
        cv2.ellipse(img,(x2-r,y2-r),(r,r),  0,0,90,color,2)


def progress_bar(img, x, y, w, h, pct, color_fill, color_bg=(55,55,75)):
    rounded_rect(img, x, y, x+w, y+h, color_bg, r=h//2)
    fill = max(h, int(pct/100*w))
    fill = min(fill, w)
    rounded_rect(img, x, y, x+fill, y+h, color_fill, r=h//2)

# ============================================
# DESSIN COMPOSANTS
# ============================================

def draw_left_panel(frame, score, back_angle, head_fwd, shoulder_tilt, fps, good_pct):
    pw, ph = 310, 430
    px, py = 16, 16
    rounded_rect(frame, px, py, px+pw, py+ph, COLOR_PANEL_BG, r=16, alpha=0.88)

    # Titre
    cv2.putText(frame, "POSTUREGHOST", (px+16, py+30),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, COLOR_GHOST, 1, cv2.LINE_AA)
    cv2.line(frame, (px+16, py+40), (px+pw-16, py+40), (60,60,80), 1)

    # Score
    status, s_color = get_status(score)
    cv2.putText(frame, str(score), (px+16, py+108),
                cv2.FONT_HERSHEY_DUPLEX, 2.9, s_color, 3, cv2.LINE_AA)
    cv2.putText(frame, "%", (px+148, py+90),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, s_color, 2, cv2.LINE_AA)
    cv2.putText(frame, status, (px+16, py+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, s_color, 1, cv2.LINE_AA)
    progress_bar(frame, px+16, py+144, 278, 14, score, s_color)

    cv2.line(frame, (px+16, py+172), (px+pw-16, py+172), (60,60,80), 1)

    # Métriques
    def metric(label, val_str, badness, y0):
        bc = COLOR_GREEN if badness < 30 else (COLOR_ORANGE if badness < 65 else COLOR_RED)
        cv2.putText(frame, label,   (px+16, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (175,175,200), 1, cv2.LINE_AA)
        cv2.putText(frame, val_str, (px+210, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_WHITE, 1, cv2.LINE_AA)
        progress_bar(frame, px+16, y0+5, 278, 7, min(100, badness), bc)

    metric("Inclinaison dos",    f"{back_angle:.1f} deg",   min(100, back_angle*1.8),   py+190)
    metric("Tete en avant",      f"{abs(head_fwd):.0f} px", min(100, abs(head_fwd)*0.9),py+230)
    metric("Asymetrie epaules",  f"{shoulder_tilt:.0f} px", min(100, shoulder_tilt*2.0),py+270)

    cv2.line(frame, (px+16, py+296), (px+pw-16, py+296), (60,60,80), 1)

    # Stats session
    cv2.putText(frame, "Bonne posture session", (px+16, py+318),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150,150,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{good_pct:.0f}%", (px+220, py+318),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_CYAN, 1, cv2.LINE_AA)
    progress_bar(frame, px+16, py+326, 278, 10, good_pct, COLOR_CYAN)

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS", (px+240, py+ph-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90,90,120), 1, cv2.LINE_AA)


def draw_ghost_skeleton(frame, landmarks, frame_shape, posture_score):
    if frame is None:
        return frame
    h, w, _ = frame_shape
    ox = int(w * 0.52)

    ghost_pts = {}
    for idx, lm in enumerate(landmarks):
        gx = int(lm.x * w * 0.40 + ox)
        gy = int(lm.y * h)
        # Correction posture idéale : redresser dos + reculer tête
        if idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                   mp_pose.PoseLandmark.RIGHT_SHOULDER.value]:
            gy -= int(h * 0.03)
        if idx in [mp_pose.PoseLandmark.LEFT_EAR.value,
                   mp_pose.PoseLandmark.RIGHT_EAR.value]:
            gy -= int(h * 0.04)
            gx -= int(w * 0.015)
        if idx in [mp_pose.PoseLandmark.LEFT_HIP.value,
                   mp_pose.PoseLandmark.RIGHT_HIP.value]:
            gy -= int(h * 0.01)
        ghost_pts[idx] = (gx, gy)

    for (si, ei) in mp_pose.POSE_CONNECTIONS:
        if si in ghost_pts and ei in ghost_pts:
            cv2.line(frame, ghost_pts[si], ghost_pts[ei], COLOR_GHOST_BONE, 3, cv2.LINE_AA)
    for pt in ghost_pts.values():
        cv2.circle(frame, pt, 5, COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 3, COLOR_GHOST, -1, cv2.LINE_AA)

    lx = int(w * 0.73)
    cv2.putText(frame, "OBJECTIF",      (lx-38, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_GHOST, 1, cv2.LINE_AA)
    cv2.putText(frame, "Posture ideale",(lx-50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,180,255), 1, cv2.LINE_AA)

    if posture_score < WARNING_THRESHOLD:
        ax, ay = int(w*0.61), int(h*0.45)
        cv2.arrowedLine(frame, (ax, ay), (ax+80, ay), COLOR_CYAN, 2,
                        cv2.LINE_AA, tipLength=0.35)
        cv2.putText(frame, "Copiez-moi !", (ax-10, ay-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)
    return frame


def draw_user_skeleton(frame, landmarks):
    if frame is None:
        return
    h, w = frame.shape[0], frame.shape[1]
    for (si, ei) in mp_pose.POSE_CONNECTIONS:
        cv2.line(frame,
                 (int(landmarks[si].x*w), int(landmarks[si].y*h)),
                 (int(landmarks[ei].x*w), int(landmarks[ei].y*h)),
                 COLOR_GREEN, 2, cv2.LINE_AA)
    for lm in landmarks:
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv2.circle(frame, (cx,cy), 4, COLOR_GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_top_alert(frame, score, status):
    if frame is None:
        return frame
    h, w = frame.shape[0], frame.shape[1]
    if score < WARNING_THRESHOLD:
        color = COLOR_RED if score < CRITICAL_THRESHOLD else COLOR_ORANGE
        msg   = ("DANGER ! Redressez-vous IMMEDIATEMENT"
                 if score < CRITICAL_THRESHOLD
                 else "Mauvaise posture — Suivez le fantome")
        rounded_rect(frame, w//2-300, 10, w//2+300, 52, color, r=10, alpha=0.78)
        cv2.putText(frame, msg, (w//2-240, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, COLOR_WHITE, 1, cv2.LINE_AA)
    if score < CRITICAL_THRESHOLD and int(time.time()*3) % 2:
        cv2.rectangle(frame, (4,4), (w-4,h-4), COLOR_RED, 3)
    return frame


def draw_ghost_zone_bg(frame):
    if frame is None:
        return frame
    h, w = frame.shape[0], frame.shape[1]
    rounded_rect(frame, int(w*0.50), 8, w-8, h-8,
                 (22, 18, 38), r=14, alpha=0.45)
    return frame


def draw_motivation(frame, last_scores):
    if frame is None or len(last_scores) < 10:
        return
    h, w = frame.shape[0], frame.shape[1]
    trend = last_scores[-1] - last_scores[-10]
    if trend > 8:
        txt, col = "Super progression !", COLOR_GREEN
    elif trend > 3:
        txt, col = "Continuez comme ca !", COLOR_CYAN
    elif trend < -8:
        txt, col = "Attention a votre dos", COLOR_ORANGE
    else:
        return
    tw = len(txt) * 9
    bx = w - tw - 40
    by = h - 40
    rounded_rect(frame, bx-10, by-20, bx+tw+10, by+12, COLOR_PANEL_BG, r=8, alpha=0.82)
    cv2.putText(frame, txt, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)


def draw_controls(frame):
    if frame is None:
        return
    cv2.putText(frame, "Q : Quitter    R : Reset    S : Capture",
                (20, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90,90,120), 1, cv2.LINE_AA)

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 55)
    print("  POSTUREGHOST  -  Coach posture IA")
    print("=" * 55)
    print("  Q : Quitter    R : Reset stats    S : Capture")
    print("=" * 55)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la webcam.")
        return

    os.makedirs("screenshots", exist_ok=True)

    frame_count       = 0
    fps               = 30.0
    prev_time         = time.time()
    last_scores       = []
    good_posture_time = 0.0
    total_time        = 0.0

    # Valeurs affichées (mises à jour à chaque détection)
    last_score    = 0
    last_back     = 0.0
    last_head_fwd = 0.0
    last_shoulder = 0.0
    last_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Erreur lecture webcam.")
            break

        frame = cv2.flip(frame, 1)

        now     = time.time()
        elapsed = now - prev_time
        fps     = 0.9 * fps + 0.1 / elapsed if elapsed > 0 else fps
        prev_time = now
        frame_count += 1

        # --- Traitement pose ---
        if frame_count % FRAME_SKIP == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            frame   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if frame is None:
                continue

            if results.pose_landmarks:
                last_landmarks = results.pose_landmarks.landmark
                last_score, last_back, last_head_fwd, last_shoulder = \
                    calculate_posture_score(last_landmarks, frame.shape)

                last_scores.append(last_score)
                if len(last_scores) > 60:
                    last_scores.pop(0)

                dt = 1.0 / fps if fps > 0 else 0
                total_time += dt
                if last_score >= GOOD_POSTURE_THRESHOLD:
                    good_posture_time += dt

        # --- Dessin ---
        if last_landmarks is not None:
            draw_user_skeleton(frame, last_landmarks)
            frame = draw_ghost_zone_bg(frame)
            frame = draw_ghost_skeleton(frame, last_landmarks, frame.shape, last_score)
        else:
            h, w = frame.shape[0], frame.shape[1]
            rounded_rect(frame, w//2-220, h//2-35, w//2+220, h//2+35,
                         COLOR_PANEL_BG, r=12, alpha=0.85)
            cv2.putText(frame, "Placez-vous devant la webcam",
                        (w//2-195, h//2+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)

        good_pct = (good_posture_time / total_time * 100) if total_time > 0 else 0
        status, _ = get_status(last_score)

        draw_left_panel(frame, last_score, last_back, last_head_fwd,
                        last_shoulder, fps, good_pct)
        frame = draw_top_alert(frame, last_score, status)
        draw_motivation(frame, last_scores)
        draw_controls(frame)

        cv2.imshow("PostureGhost", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            good_posture_time = 0.0
            total_time        = 0.0
            last_scores.clear()
            print("Stats reinitialisees.")
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"screenshots/posture_{ts}.png"
            cv2.imwrite(fn, frame)
            print(f"Capture : {fn}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print("\n" + "=" * 55)
    print("  SESSION TERMINEE")
    print("=" * 55)
    if total_time > 0:
        print(f"  Duree         : {total_time:.0f} sec")
        print(f"  Bonne posture : {good_posture_time:.0f} sec")
        print(f"  Pourcentage   : {good_posture_time/total_time*100:.1f} %")
    print("=" * 55)


if __name__ == "__main__":
    main()