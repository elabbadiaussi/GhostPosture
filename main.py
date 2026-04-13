"""
POSTUREGHOST - Coach posture IA avec fantôme + Voice Coach + Gamification
Version 2.0 - Améliorations visuelles et fonctionnelles
Challenge: Wellness & Agent Challenge GIEW 2026

Nouveautés v2.0:
  - Panneau gauche redesigné avec anneau de score animé
  - Mode calibration : capture ta propre posture idéale comme fantôme
  - Mini-graphe historique du score en temps réel
  - Rappels de pause configurable (minuterie Pomodoro)
  - Heatmap des articulations sous tension
  - Export CSV automatique de la session
  - Overlay de respiration guidée (inspire / expire)
  - Détection de position assise vs debout
  - Indicateur de tendance (amélioration / dégradation)
  - Affichage HUD plus propre avec transitions de couleur fluides
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import threading
import pyttsx3
from datetime import datetime
from collections import deque

# Gamification module (inchangé)
try:
    from gamification import GamificationManager
    GAMIFICATION_AVAILABLE = True
except ImportError:
    GAMIFICATION_AVAILABLE = False
    print("  [AVERT] Module gamification introuvable - désactivé.")

# ============================================
# CONFIGURATION
# ============================================

FRAME_SKIP          = 1
RESOLUTION_WIDTH    = 1280
RESOLUTION_HEIGHT   = 720

GOOD_POSTURE_THRESHOLD = 75
WARNING_THRESHOLD      = 55
CRITICAL_THRESHOLD     = 35

# Minuterie de pause (secondes). 0 = désactivé.
BREAK_REMINDER_SEC = 25 * 60   # 25 min par défaut (Pomodoro)

# Export CSV automatique
EXPORT_CSV = True

# Couleurs (BGR)
COLOR_GREEN      = (0, 220, 100)
COLOR_RED        = (50, 50, 230)
COLOR_ORANGE     = (0, 140, 255)
COLOR_WHITE      = (255, 255, 255)
COLOR_CYAN       = (220, 220, 0)
COLOR_GHOST      = (255, 200, 180)
COLOR_GHOST_BONE = (200, 170, 255)
COLOR_PANEL_BG   = (28, 24, 42)
COLOR_ACCENT     = (180, 100, 255)   # violet accent
COLOR_GOLD       = (0, 200, 255)     # badge gold

# Palette score (interpolée selon score 0-100)
def score_color(score):
    """Renvoie une couleur BGR interpolée entre rouge, orange et vert."""
    if score >= GOOD_POSTURE_THRESHOLD:
        return COLOR_GREEN
    elif score >= WARNING_THRESHOLD:
        t = (score - WARNING_THRESHOLD) / (GOOD_POSTURE_THRESHOLD - WARNING_THRESHOLD)
        r = int(COLOR_ORANGE[0] + t * (COLOR_GREEN[0] - COLOR_ORANGE[0]))
        g = int(COLOR_ORANGE[1] + t * (COLOR_GREEN[1] - COLOR_ORANGE[1]))
        b = int(COLOR_ORANGE[2] + t * (COLOR_GREEN[2] - COLOR_ORANGE[2]))
        return (r, g, b)
    elif score >= CRITICAL_THRESHOLD:
        t = (score - CRITICAL_THRESHOLD) / (WARNING_THRESHOLD - CRITICAL_THRESHOLD)
        r = int(COLOR_RED[0] + t * (COLOR_ORANGE[0] - COLOR_RED[0]))
        g = int(COLOR_RED[1] + t * (COLOR_ORANGE[1] - COLOR_RED[1]))
        b = int(COLOR_RED[2] + t * (COLOR_ORANGE[2] - COLOR_RED[2]))
        return (r, g, b)
    else:
        return COLOR_RED

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
# VOICE COACH
# ============================================

class VoiceCoach:
    def __init__(self, lang='fr'):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 155)
        self.engine.setProperty('volume', 0.9)
        self._set_language(lang)
        self._speaking  = False
        self._last_spoken = 0
        self._cooldown  = 8.0
        self.muted      = False

    def _set_language(self, lang):
        try:
            voices = self.engine.getProperty('voices')
            for v in voices:
                if lang in str(v.languages).lower() or lang in v.id.lower():
                    self.engine.setProperty('voice', v.id)
                    break
        except Exception:
            pass

    def _speak_blocking(self, text):
        self._speaking = True
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception:
            pass
        self._speaking = False

    def say(self, text, force=False):
        if self.muted:
            return
        now = time.time()
        if self._speaking:
            return
        if not force and (now - self._last_spoken) < self._cooldown:
            return
        self._last_spoken = now
        t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        t.start()

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass


def get_voice_alert(score, back_angle, head_fwd, shoulder_tilt):
    if score >= GOOD_POSTURE_THRESHOLD:
        return None, False
    back_badness     = min(100, back_angle * 1.8)
    head_badness     = min(100, abs(head_fwd) * 0.9)
    shoulder_badness = min(100, shoulder_tilt * 2.0)
    worst = max(back_badness, head_badness, shoulder_badness)
    if score < CRITICAL_THRESHOLD:
        return "Danger ! Redressez-vous immédiatement.", True
    if worst == back_badness:
        return "Attention, votre dos est penché. Redressez-vous.", False
    elif worst == head_badness:
        return "Votre tête est trop en avant. Rentrez le menton.", False
    else:
        return "Vos épaules sont déséquilibrées. Alignez-les.", False

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

    ls = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    lh = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
    rh = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
    le = pt(mp_pose.PoseLandmark.LEFT_EAR.value)
    re = pt(mp_pose.PoseLandmark.RIGHT_EAR.value)

    back_angle = (
        calculate_angle(ls, lh, [lh[0], lh[1] - 100]) +
        calculate_angle(rs, rh, [rh[0], rh[1] - 100])
    ) / 2

    ear_mid  = [(le[0]+re[0])/2, (le[1]+re[1])/2]
    sh_mid   = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
    head_fwd = ear_mid[0] - sh_mid[0]

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


def detect_sitting(landmarks, frame_shape):
    """Heuristique simple : si la hanche est proche du bas du cadre."""
    h, _ = frame_shape[:2]
    lh_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    rh_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    hip_y = ((lh_y + rh_y) / 2) * h
    return hip_y > h * 0.60

# ============================================
# HELPERS DESSIN
# ============================================

def rounded_rect(img, x1, y1, x2, y2, color, r=12, fill=True, alpha=1.0):
    if img is None:
        return
    x1, y1, x2, y2, r = int(x1), int(y1), int(x2), int(y2), int(r)
    if alpha < 1.0:
        ov = img.copy()
        rounded_rect(ov, x1, y1, x2, y2, color, r, fill, 1.0)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
        return
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


def progress_bar(img, x, y, w, h, pct, color_fill, color_bg=(50, 45, 70)):
    rounded_rect(img, x, y, x+w, y+h, color_bg, r=h//2)
    fill = max(h, int(pct/100*w))
    fill = min(fill, w)
    rounded_rect(img, x, y, x+fill, y+h, color_fill, r=h//2)


def draw_arc_ring(img, cx, cy, radius, thickness, pct, color, bg_color=(55, 50, 75)):
    """Dessine un anneau de progression type donut."""
    start_angle = -90
    end_angle   = -90 + int(360 * pct / 100)
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, 0, 360, bg_color, thickness, cv2.LINE_AA)
    if pct > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)


def draw_score_ring(frame, cx, cy, score):
    """Anneau animé centré sur (cx, cy) avec score + label."""
    col = score_color(score)
    draw_arc_ring(frame, cx, cy, 56, 10, score, col)
    # Pulsation danger
    if score < CRITICAL_THRESHOLD and int(time.time() * 4) % 2:
        draw_arc_ring(frame, cx, cy, 64, 3, 100, COLOR_RED)
    s_str = str(score)
    tw = len(s_str) * 14
    cv2.putText(frame, s_str, (cx - tw//2 + 2, cy + 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, col, 2, cv2.LINE_AA)
    cv2.putText(frame, "%", (cx + tw//2 - 2, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def draw_mini_graph(frame, scores, x, y, w, h, color=(130, 100, 220)):
    """Mini-graphe du score sur les 90 dernières frames."""
    if len(scores) < 2:
        return
    rounded_rect(frame, x, y, x+w, y+h, (38, 33, 58), r=6, alpha=0.85)
    # Lignes de grille subtiles
    for thresh, c in [(GOOD_POSTURE_THRESHOLD, COLOR_GREEN),
                      (WARNING_THRESHOLD, COLOR_ORANGE),
                      (CRITICAL_THRESHOLD, COLOR_RED)]:
        gy = y + h - int(thresh / 100 * h)
        cv2.line(frame, (x+2, gy), (x+w-2, gy), c, 1, cv2.LINE_AA)
    # Courbe
    pts = []
    n = len(scores)
    for i, s in enumerate(scores):
        px = x + int(i / max(n-1, 1) * w)
        py = y + h - int(s / 100 * h)
        pts.append((px, py))
    for i in range(len(pts)-1):
        col = score_color(scores[i])
        cv2.line(frame, pts[i], pts[i+1], col, 2, cv2.LINE_AA)


def draw_breathing_guide(frame, w, h, session_time):
    """Indicateur de respiration guidée en bas à droite."""
    cycle = 8.0  # 4s inspire + 4s expire
    phase = session_time % cycle
    if phase < 4.0:
        label = "INSPIREZ"
        progress = phase / 4.0
        col = (200, 240, 255)
    else:
        label = "EXPIREZ"
        progress = (phase - 4.0) / 4.0
        col = (150, 200, 170)
    bx, by = w - 135, h - 95
    radius = 22
    draw_arc_ring(frame, bx, by, radius, 5, int(progress * 100), col, (40, 40, 60))
    cv2.putText(frame, label, (bx - 28, by + radius + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    cv2.putText(frame, "Respiration", (bx - 34, by - radius - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110, 110, 140), 1, cv2.LINE_AA)


def draw_break_reminder(frame, w, h, time_until_break, total_break_sec):
    """Barre de progression de la pause."""
    if total_break_sec <= 0:
        return
    elapsed = total_break_sec - time_until_break
    pct = min(100, elapsed / total_break_sec * 100)
    bx, by = 340, h - 46
    bar_w = 280
    # Fond
    rounded_rect(frame, bx, by, bx + bar_w, by + 13, (38, 33, 58), r=6, alpha=0.85)
    # Remplissage (couleur vire vers orange en fin)
    col = COLOR_CYAN if pct < 70 else (COLOR_ORANGE if pct < 90 else COLOR_RED)
    fill = max(6, int(pct / 100 * bar_w))
    rounded_rect(frame, bx, by, bx + fill, by + 13, col, r=6)
    mins = int(time_until_break // 60)
    secs = int(time_until_break % 60)
    cv2.putText(frame, f"Pause dans {mins:02d}:{secs:02d}",
                (bx + 6, by + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_WHITE, 1, cv2.LINE_AA)
    if time_until_break <= 0:
        # Toast pause !
        tw = 220
        tx = w // 2 - tw // 2
        rounded_rect(frame, tx, h//2 - 30, tx + tw, h//2 + 30, (50, 140, 80), r=12, alpha=0.92)
        cv2.putText(frame, "Pause ! Levez-vous 5 min", (tx + 8, h//2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_trend_indicator(frame, scores, x, y):
    """Flèche de tendance (+/-) selon les 20 dernières frames."""
    if len(scores) < 20:
        return
    delta = scores[-1] - scores[-20]
    if delta > 5:
        symbol, col = chr(0x2191), COLOR_GREEN   # ↑
    elif delta < -5:
        symbol, col = chr(0x2193), COLOR_RED     # ↓
    else:
        symbol, col = chr(0x2192), (130, 130, 160)  # →
    cv2.putText(frame, symbol, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{delta:+.0f}", (x+18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

# ============================================
# DESSIN SQUELETTES
# ============================================

def _joint_strain(lm_idx, back_angle, head_fwd, shoulder_tilt):
    """Retourne une valeur 0-1 de 'tension' pour un landmark donné."""
    back_bad  = min(1.0, back_angle / 55.0)
    head_bad  = min(1.0, abs(head_fwd) / 110.0)
    sh_bad    = min(1.0, shoulder_tilt / 50.0)
    spine_joints = {
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
    }
    neck_joints = {
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.NOSE.value,
    }
    if lm_idx in spine_joints:
        return max(back_bad, sh_bad)
    if lm_idx in neck_joints:
        return head_bad
    return 0.0


def draw_user_skeleton(frame, landmarks, back_angle=0, head_fwd=0, shoulder_tilt=0):
    """Squelette avec heatmap couleur selon tension des articulations."""
    if frame is None:
        return
    h, w = frame.shape[0], frame.shape[1]
    for (si, ei) in mp_pose.POSE_CONNECTIONS:
        strain = (_joint_strain(si, back_angle, head_fwd, shoulder_tilt) +
                  _joint_strain(ei, back_angle, head_fwd, shoulder_tilt)) / 2
        # Interpolation couleur vert → rouge selon tension
        r = int(0   + strain * 180)
        g = int(220 - strain * 190)
        b = int(100 - strain * 100)
        col = (r, g, b)
        cv2.line(frame,
                 (int(landmarks[si].x*w), int(landmarks[si].y*h)),
                 (int(landmarks[ei].x*w), int(landmarks[ei].y*h)),
                 col, 2, cv2.LINE_AA)
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x*w), int(lm.y*h)
        strain = _joint_strain(idx, back_angle, head_fwd, shoulder_tilt)
        r = int(0   + strain * 200)
        g = int(220 - strain * 200)
        b = int(100 - strain * 100)
        col = (r, g, b)
        cv2.circle(frame, (cx, cy), 5, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_ghost_skeleton(frame, landmarks, calibrated_landmarks, frame_shape, posture_score):
    """Fantôme avec support des landmarks calibrés comme référence."""
    if frame is None:
        return frame
    h, w, _ = frame_shape
    ox = int(w * 0.52)

    source = calibrated_landmarks if calibrated_landmarks else landmarks

    ghost_pts = {}
    for idx, lm in enumerate(source):
        gx = int(lm.x * w * 0.40 + ox)
        gy = int(lm.y * h)
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
    ghost_label = "CALIBRE" if calibrated_landmarks else "OBJECTIF"
    ghost_sub   = "Ta posture ideale" if calibrated_landmarks else "Posture ideale"
    cv2.putText(frame, ghost_label, (lx-38, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_GHOST, 1, cv2.LINE_AA)
    cv2.putText(frame, ghost_sub,   (lx-50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,180,255), 1, cv2.LINE_AA)

    if posture_score < WARNING_THRESHOLD:
        ax, ay = int(w*0.61), int(h*0.45)
        cv2.arrowedLine(frame, (ax, ay), (ax+80, ay), COLOR_CYAN, 2,
                        cv2.LINE_AA, tipLength=0.35)
        cv2.putText(frame, "Copiez-moi !", (ax-10, ay-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)
    return frame

# ============================================
# PANNEAUX HUD
# ============================================

def draw_left_panel(frame, score, back_angle, head_fwd, shoulder_tilt,
                    fps, good_pct, muted, scores_history, sitting):
    pw, ph = 320, 490
    px, py = 16, 16
    rounded_rect(frame, px, py, px+pw, py+ph, COLOR_PANEL_BG, r=18, alpha=0.90)

    # En-tête
    cv2.putText(frame, "POSTUREGHOST", (px+16, py+30),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, COLOR_ACCENT, 1, cv2.LINE_AA)

    # Badge position assise/debout
    pos_label = "Assis" if sitting else "Debout"
    pos_col   = (130, 200, 255) if sitting else (100, 230, 130)
    rounded_rect(frame, px+200, py+14, px+300, py+34, (45, 38, 70), r=8)
    cv2.putText(frame, pos_label, (px+214, py+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, pos_col, 1, cv2.LINE_AA)

    cv2.line(frame, (px+16, py+42), (px+pw-16, py+42), (55, 48, 80), 1)

    # Anneau score + label statut
    ring_cx, ring_cy = px + 74, py + 120
    draw_score_ring(frame, ring_cx, ring_cy, score)
    status, s_color = get_status(score)
    cv2.putText(frame, status, (ring_cx - 28, ring_cy + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, s_color, 1, cv2.LINE_AA)

    # Tendance
    draw_trend_indicator(frame, scores_history, px+160, ring_cy - 10)

    # Mini graphe
    draw_mini_graph(frame, list(scores_history), px+148, ring_cy - 55, 155, 100)

    cv2.line(frame, (px+16, py+200), (px+pw-16, py+200), (55, 48, 80), 1)

    # Métriques
    def metric(label, val_str, badness, y0):
        bc = COLOR_GREEN if badness < 30 else (COLOR_ORANGE if badness < 65 else COLOR_RED)
        cv2.putText(frame, label,   (px+16, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (175, 175, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, val_str, (px+220, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
        progress_bar(frame, px+16, y0+5, 288, 7, min(100, badness), bc)

    metric("Inclinaison dos",   f"{back_angle:.1f} deg",   min(100, back_angle*1.8),    py+220)
    metric("Tete en avant",     f"{abs(head_fwd):.0f} px", min(100, abs(head_fwd)*0.9), py+260)
    metric("Asymetrie epaules", f"{shoulder_tilt:.0f} px", min(100, shoulder_tilt*2.0), py+300)

    cv2.line(frame, (px+16, py+328), (px+pw-16, py+328), (55, 48, 80), 1)

    # Bonne posture session
    cv2.putText(frame, "Bonne posture session", (px+16, py+350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 150, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{good_pct:.0f}%", (px+232, py+350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_CYAN, 1, cv2.LINE_AA)
    progress_bar(frame, px+16, py+358, 288, 10, good_pct, COLOR_CYAN)

    cv2.line(frame, (px+16, py+382), (px+pw-16, py+382), (55, 48, 80), 1)

    # Mute + FPS
    mute_color = COLOR_RED if muted else COLOR_GREEN
    cv2.circle(frame, (px+26, py+404), 7, mute_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "Son COUPE" if muted else "Son ACTIF",
                (px+40, py+409), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mute_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:.0f} FPS", (px+244, py+ph-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 110), 1, cv2.LINE_AA)


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
    rounded_rect(frame, int(w*0.50), 8, w-8, h-8, (20, 16, 36), r=14, alpha=0.45)
    return frame


def draw_calibration_overlay(frame, countdown):
    """Affiche un compte à rebours de calibration en overlay."""
    h, w = frame.shape[:2]
    rounded_rect(frame, w//2-240, h//2-55, w//2+240, h//2+55,
                 (30, 25, 50), r=16, alpha=0.92)
    cv2.putText(frame, "CALIBRATION",
                (w//2-90, h//2-20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Adoptez votre meilleure posture... {countdown}",
                (w//2-200, h//2+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    # Anneau de progression
    draw_arc_ring(frame, w//2+185, h//2, 20, 4,
                  int((3 - countdown) / 3 * 100), COLOR_ACCENT)


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
    by = h - 58
    rounded_rect(frame, bx-10, by-20, bx+tw+10, by+12, COLOR_PANEL_BG, r=8, alpha=0.82)
    cv2.putText(frame, txt, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)


def draw_gamification_panel(frame, gm, new_badge=None):
    h, w = frame.shape[:2]
    summ = gm.summary()
    bar_x, bar_y = 340, h - 28
    bar_w = 300
    pct = min(1.0, summ["xp"] / max(1, summ["next_xp"]))
    rounded_rect(frame, bar_x, bar_y, bar_x + bar_w, bar_y + 16, (38, 32, 65), r=8, alpha=0.85)
    fill_w = max(16, int(pct * bar_w))
    rounded_rect(frame, bar_x, bar_y, bar_x + fill_w, bar_y + 16, (130, 100, 220), r=8)
    cv2.putText(frame, f"Niv. {summ['level']}  {summ['xp']} XP",
                (bar_x + 8, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)
    sx = bar_x + bar_w + 12
    streak_col = COLOR_CYAN if summ["streak"] >= 3 else (130, 130, 160)
    cv2.putText(frame, f"Serie: {summ['streak']}j",  (sx, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, streak_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{summ['badges']} badges", (sx + 90, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 140, 210), 1, cv2.LINE_AA)
    if new_badge:
        msg = f"{new_badge['icon']} {new_badge['name']}  +{new_badge['xp']} XP"
        tw  = len(msg) * 10
        tx  = w // 2 - tw // 2
        rounded_rect(frame, tx-12, 62, tx+tw+12, 94, (80, 60, 160), r=12, alpha=0.92)
        cv2.putText(frame, msg, (tx, 84),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_controls(frame):
    if frame is None:
        return
    h = frame.shape[0]
    cv2.putText(frame,
                "Q:Quit  R:Reset  S:Capture  M:Mute  C:Calibrer  B:Respiration  P:Pause ON/OFF",
                (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 115), 1, cv2.LINE_AA)

# ============================================
# EXPORT CSV
# ============================================

class SessionLogger:
    def __init__(self):
        self.rows = []
        self.start_time = time.time()

    def log(self, score, back, head, shoulder, sitting):
        elapsed = time.time() - self.start_time
        self.rows.append({
            "t_sec": round(elapsed, 1),
            "score": score,
            "back_angle": round(back, 2),
            "head_fwd":   round(head, 2),
            "shoulder_tilt": round(shoulder, 2),
            "sitting": int(sitting),
        })

    def save(self):
        if not self.rows:
            return
        os.makedirs("sessions", exist_ok=True)
        fn = f"sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fn, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"  Export CSV : {fn}")
        return fn

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("  POSTUREGHOST v2.0  -  Coach IA + Voix + XP + Graphes")
    print("=" * 60)
    print("  Q : Quitter    R : Reset stats    S : Capture")
    print("  M : Mute/Unmute    C : Calibrer posture")
    print("  B : Toggle respiration    P : Pause ON/OFF")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la webcam.")
        return

    os.makedirs("screenshots", exist_ok=True)

    voice = VoiceCoach(lang='fr')
    print("  Coach vocal initialise.")

    if GAMIFICATION_AVAILABLE:
        gm = GamificationManager()
        gm.start_session()
        print(f"  Niveau : {gm.summary()['level']}  |  XP : {gm.stats['xp']}")
    else:
        gm = None

    logger = SessionLogger() if EXPORT_CSV else None

    frame_count       = 0
    fps               = 30.0
    prev_time         = time.time()
    session_start     = time.time()
    scores_history    = deque(maxlen=90)
    good_posture_time = 0.0
    total_time        = 0.0

    last_score      = 0
    last_back       = 0.0
    last_head_fwd   = 0.0
    last_shoulder   = 0.0
    last_landmarks  = None
    sitting         = False

    # Ghost / calibration
    calibrated_landmarks = None
    calibrating          = False
    calib_start          = 0.0
    CALIB_DURATION       = 3.0

    active_badge      = None
    badge_show_until  = 0.0

    # Minuterie de pause
    break_timer_active   = BREAK_REMINDER_SEC > 0
    time_of_last_break   = time.time()

    # Respiration guidée
    show_breathing = True

    print("=" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Erreur lecture webcam.")
            break

        frame = cv2.flip(frame, 1)
        now     = time.time()
        elapsed = now - prev_time
        fps     = 0.9 * fps + 0.1 / elapsed if elapsed > 0 else fps
        prev_time   = now
        session_time = now - session_start
        frame_count += 1

        # ---- Traitement pose ----
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
                sitting = detect_sitting(last_landmarks, frame.shape)

                scores_history.append(last_score)

                dt = 1.0 / fps if fps > 0 else 0
                total_time += dt
                if last_score >= GOOD_POSTURE_THRESHOLD:
                    good_posture_time += dt
                    if gm:
                        gm.on_good_frame(last_score)

                if gm:
                    gm.on_score(last_score)
                    earned = gm.pop_new_badges()
                    if earned:
                        active_badge     = earned[0]
                        badge_show_until = now + 3.0
                        voice.say(f"Badge débloqué : {earned[0]['name']} !", force=True)

                alert, force = get_voice_alert(
                    last_score, last_back, last_head_fwd, last_shoulder)
                if alert:
                    voice.say(alert, force=force)

                if logger and frame_count % 15 == 0:
                    logger.log(last_score, last_back, last_head_fwd, last_shoulder, sitting)

                # Calibration : capture du fantôme après décompte
                if calibrating and (now - calib_start) >= CALIB_DURATION:
                    calibrated_landmarks = list(last_landmarks)
                    calibrating = False
                    voice.say("Calibration enregistrée ! Ce sera votre posture de référence.", force=True)
                    print("  Calibration OK.")

        # Badge toast timeout
        if active_badge and now > badge_show_until:
            active_badge = None

        # Pause reminder
        time_until_break = 0
        if break_timer_active and BREAK_REMINDER_SEC > 0:
            time_until_break = max(0, BREAK_REMINDER_SEC - (now - time_of_last_break))
            if time_until_break == 0 and now - time_of_last_break > BREAK_REMINDER_SEC:
                voice.say("C'est l'heure de faire une pause ! Levez-vous et bougez.", force=True)
                time_of_last_break = now  # reset

        # ---- Dessin ----
        if last_landmarks is not None:
            draw_user_skeleton(frame, last_landmarks, last_back, last_head_fwd, last_shoulder)
            frame = draw_ghost_zone_bg(frame)
            frame = draw_ghost_skeleton(frame, last_landmarks, calibrated_landmarks,
                                        frame.shape, last_score)
        else:
            h_f, w_f = frame.shape[0], frame.shape[1]
            rounded_rect(frame, w_f//2-220, h_f//2-35, w_f//2+220, h_f//2+35,
                         COLOR_PANEL_BG, r=12, alpha=0.85)
            cv2.putText(frame, "Placez-vous devant la webcam",
                        (w_f//2-195, h_f//2+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)

        good_pct = (good_posture_time / total_time * 100) if total_time > 0 else 0
        status, _ = get_status(last_score)
        h_f, w_f = frame.shape[0], frame.shape[1]

        draw_left_panel(frame, last_score, last_back, last_head_fwd,
                        last_shoulder, fps, good_pct, voice.muted,
                        scores_history, sitting)
        frame = draw_top_alert(frame, last_score, status)
        draw_motivation(frame, list(scores_history))

        if gm:
            draw_gamification_panel(frame, gm, active_badge)

        if break_timer_active and BREAK_REMINDER_SEC > 0:
            draw_break_reminder(frame, w_f, h_f, time_until_break, BREAK_REMINDER_SEC)

        if show_breathing:
            draw_breathing_guide(frame, w_f, h_f, session_time)

        if calibrating:
            countdown = max(0, int(CALIB_DURATION - (now - calib_start)) + 1)
            draw_calibration_overlay(frame, countdown)

        draw_controls(frame)

        cv2.imshow("PostureGhost v2.0", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            good_posture_time = 0.0
            total_time        = 0.0
            scores_history.clear()
            print("Stats reinitialisees.")
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"screenshots/posture_{ts}.png"
            cv2.imwrite(fn, frame)
            print(f"Capture : {fn}")
        elif key == ord('m'):
            voice.muted = not voice.muted
            print("Son : COUPE" if voice.muted else "Son : ACTIF")
        elif key == ord('c'):
            # Démarrer calibration
            if last_landmarks is not None:
                calibrating  = True
                calib_start  = time.time()
                print("  Calibration démarrée — gardez la meilleure posture...")
            else:
                print("  Aucun squelette détecté, impossible de calibrer.")
        elif key == ord('b'):
            show_breathing = not show_breathing
        elif key == ord('p'):
            break_timer_active = not break_timer_active
            print(f"Rappel pause : {'ON' if break_timer_active else 'OFF'}")

    # ---- Cleanup & session summary ----
    if gm:
        gm.end_session(good_posture_time, last_score)
    voice.stop()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    if logger:
        logger.save()

    print("\n" + "=" * 60)
    print("  SESSION TERMINEE")
    print("=" * 60)
    if total_time > 0:
        print(f"  Duree         : {total_time:.0f} sec")
        print(f"  Bonne posture : {good_posture_time:.0f} sec")
        print(f"  Pourcentage   : {good_posture_time/total_time*100:.1f} %")
    if gm:
        print(f"  XP cette session : +{gm.xp_gained}")
        print(f"  Total XP         : {gm.stats['xp']}")
        print(f"  Niveau           : {gm.summary()['level']}")
        print(f"  Badges           : {gm.stats['unlocked']}")
    print("=" * 60)


if __name__ == "__main__":
    main()