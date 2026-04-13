"""
POSTUREGHOST - Coach posture IA avec fantôme + Voice Coach + Gamification
Version 3.0 - Nouvelles fonctionnalités majeures
Challenge: Wellness & Agent Challenge GIEW 2026

Nouveautés v3.0:
  - Rapport de session HTML généré automatiquement à la fin
  - Mode focus / distraction : détecte les regards hors caméra
  - Compteur de micro-pauses intelligentes (stretch reminders)
  - Panneau droit : conseils posturaux en rotation automatique
  - Détection de fatigue : score qui chute progressivement = alerte
  - Historique inter-sessions (fichier JSON persistant)
  - Affichage du temps total en bonne / mauvaise posture dans le HUD
  - Mode nuit : palette sombre réduite à partir de 20h
  - Overlay angle de cou (neck angle) en degrés sur le squelette
  - Touche H : affiche un panneau d'aide complet en overlay
  - Score moyen de session affiché en temps réel
  - Détection de haussement d'épaules (stress indicator)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import json
import threading
import pyttsx3
from datetime import datetime
from collections import deque
from pathlib import Path

try:
    from gamification import GamificationManager
    GAMIFICATION_AVAILABLE = True
except ImportError:
    GAMIFICATION_AVAILABLE = False
    print("  [AVERT] Module gamification introuvable - desactive.")

# ============================================================
# CONFIGURATION
# ============================================================

FRAME_SKIP          = 1
RESOLUTION_WIDTH    = 1280
RESOLUTION_HEIGHT   = 720

GOOD_POSTURE_THRESHOLD = 75
WARNING_THRESHOLD      = 55
CRITICAL_THRESHOLD     = 35

BREAK_REMINDER_SEC  = 25 * 60      # Pomodoro 25 min
STRETCH_INTERVAL    = 5 * 60       # Rappel stretch toutes les 5 min
EXPORT_CSV          = True
EXPORT_HTML_REPORT  = True
HISTORY_FILE        = "sessions/history.json"

# Couleurs BGR
COLOR_GREEN      = (0, 220, 100)
COLOR_RED        = (50, 50, 230)
COLOR_ORANGE     = (0, 140, 255)
COLOR_WHITE      = (255, 255, 255)
COLOR_CYAN       = (220, 220, 0)
COLOR_GHOST      = (255, 200, 180)
COLOR_GHOST_BONE = (200, 170, 255)
COLOR_PANEL_BG   = (28, 24, 42)
COLOR_ACCENT     = (180, 100, 255)
COLOR_GOLD       = (0, 200, 255)
COLOR_DARK_BG    = (18, 14, 28)    # mode nuit

# Conseils posturaux rotatifs
POSTURE_TIPS = [
    "Ecrans a hauteur des yeux",
    "Pieds a plat sur le sol",
    "Dos droit, epaules detendues",
    "Coudes a 90 degres",
    "Clavier proche du corps",
    "Regle le siege : hanches > genoux",
    "20-20-20 : pause visuelle toutes 20 min",
    "Boire de l eau regulierement",
    "Epaules en arriere, pas en avant",
    "Menton legerement rentre",
]

STRETCH_EXERCISES = [
    "Roulez les epaules 5 fois",
    "Inclinez la tete gauche/droite 10s",
    "Etirement du cou : menton vers poitrine",
    "Levez les bras au-dessus 20s",
    "Rotation du buste assis x5",
    "Pressez les omoplates 10s",
]

# ============================================================
# MEDIAPIPE
# ============================================================

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

# ============================================================
# HELPERS COULEUR
# ============================================================

def score_color(score):
    """BGR interpolee rouge -> orange -> vert."""
    if score >= GOOD_POSTURE_THRESHOLD:
        return COLOR_GREEN
    elif score >= WARNING_THRESHOLD:
        t = (score - WARNING_THRESHOLD) / (GOOD_POSTURE_THRESHOLD - WARNING_THRESHOLD)
        return tuple(int(COLOR_ORANGE[i] + t * (COLOR_GREEN[i] - COLOR_ORANGE[i])) for i in range(3))
    elif score >= CRITICAL_THRESHOLD:
        t = (score - CRITICAL_THRESHOLD) / (WARNING_THRESHOLD - CRITICAL_THRESHOLD)
        return tuple(int(COLOR_RED[i] + t * (COLOR_ORANGE[i] - COLOR_RED[i])) for i in range(3))
    return COLOR_RED


def night_mode_active():
    return datetime.now().hour >= 20 or datetime.now().hour < 6


def panel_bg():
    return COLOR_DARK_BG if night_mode_active() else COLOR_PANEL_BG

# ============================================================
# VOICE COACH
# ============================================================

class VoiceCoach:
    def __init__(self, lang='fr'):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 155)
        self.engine.setProperty('volume', 0.9)
        self._set_language(lang)
        self._speaking    = False
        self._last_spoken = 0
        self._cooldown    = 8.0
        self.muted        = False

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
        threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass


def get_voice_alert(score, back_angle, head_fwd, shoulder_tilt):
    if score >= GOOD_POSTURE_THRESHOLD:
        return None, False
    back_bad = min(100, back_angle * 1.8)
    head_bad = min(100, abs(head_fwd) * 0.9)
    sh_bad   = min(100, shoulder_tilt * 2.0)
    worst    = max(back_bad, head_bad, sh_bad)
    if score < CRITICAL_THRESHOLD:
        return "Danger ! Redressez-vous immediatement.", True
    if worst == back_bad:
        return "Votre dos est penche. Redressez-vous.", False
    elif worst == head_bad:
        return "Tete trop en avant. Rentrez le menton.", False
    else:
        return "Epaules desequilibrees. Alignez-les.", False

# ============================================================
# CALCULS
# ============================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
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
        calculate_angle(ls, lh, [lh[0], lh[1]-100]) +
        calculate_angle(rs, rh, [rh[0], rh[1]-100])
    ) / 2

    ear_mid  = [(le[0]+re[0])/2, (le[1]+re[1])/2]
    sh_mid   = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2]
    head_fwd = ear_mid[0] - sh_mid[0]

    shoulder_tilt = abs(ls[1] - rs[1])

    neck_angle = calculate_angle(le, ls, [ls[0], ls[1]-100])

    back_score     = max(0, 100 - back_angle * 1.8)
    head_score     = max(0, 100 - abs(head_fwd) * 0.9)
    shoulder_score = max(0, 100 - shoulder_tilt * 2.0)

    score = int(back_score * 0.50 + head_score * 0.30 + shoulder_score * 0.20)
    return max(0, min(100, score)), back_angle, head_fwd, shoulder_tilt, neck_angle


def get_status(score):
    if score >= GOOD_POSTURE_THRESHOLD:
        return "PARFAIT",   COLOR_GREEN
    elif score >= WARNING_THRESHOLD:
        return "ATTENTION", COLOR_ORANGE
    else:
        return "DANGER",    COLOR_RED


def detect_sitting(landmarks, frame_shape):
    h = frame_shape[0]
    lh_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    rh_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    return ((lh_y + rh_y) / 2) * h > h * 0.60


def detect_shrug(landmarks, frame_shape):
    """Epaules trop hautes = haussement de stress."""
    h = frame_shape[0]
    ls_y  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
    rs_y  = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
    lh_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
    rh_y  = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
    sh_y  = (ls_y + rs_y) / 2
    hip_y = (lh_y + rh_y) / 2
    ratio = (hip_y - sh_y) / max(1, hip_y)
    return ratio < 0.28


def detect_distraction(landmarks, frame_shape):
    """Nez trop decale = regard tourne."""
    w = frame_shape[1]
    nose_x      = landmarks[mp_pose.PoseLandmark.NOSE.value].x * w
    le_x        = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w
    re_x        = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w
    face_center = (le_x + re_x) / 2
    return abs(nose_x - face_center) > w * 0.06


def detect_fatigue(scores_history):
    """Chute continue sur 60 frames = fatigue."""
    if len(scores_history) < 60:
        return False
    recent = list(scores_history)[-60:]
    return (np.mean(recent[:30]) - np.mean(recent[30:])) > 12

# ============================================================
# HISTORIQUE INTER-SESSIONS
# ============================================================

class SessionHistory:
    def __init__(self, path=HISTORY_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"sessions": []}

    def save_session(self, duration_sec, good_pct, avg_score, best_score):
        entry = {
            "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration_sec": round(duration_sec),
            "good_pct":     round(good_pct, 1),
            "avg_score":    round(avg_score, 1),
            "best_score":   best_score,
        }
        self.data["sessions"].append(entry)
        self.data["sessions"] = self.data["sessions"][-30:]
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        return entry

    def best_streak(self):
        streak = 0
        for s in reversed(self.data["sessions"]):
            if s.get("good_pct", 0) > 60:
                streak += 1
            else:
                break
        return streak

    def all_time_avg(self):
        if not self.data["sessions"]:
            return 0
        return round(np.mean([s["avg_score"] for s in self.data["sessions"]]), 1)

# ============================================================
# RAPPORT HTML
# ============================================================

def generate_html_report(duration_sec, good_pct, avg_score, best_score,
                         scores_list, history):
    os.makedirs("sessions", exist_ok=True)
    fn = f"sessions/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    if len(scores_list) > 1:
        n      = len(scores_list)
        step   = max(1, n // 200)
        sampled = scores_list[::step]
        pts = []
        for i, s in enumerate(sampled):
            x = 10 + int(i / max(len(sampled)-1, 1) * 580)
            y = 90 - int(s / 100 * 80)
            pts.append(f"{x},{y}")
        svg_graph = (
            '<svg width="600" height="100" style="background:#1a1628;border-radius:8px">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="#a06cff" stroke-width="2"/>'
            '</svg>'
        )
    else:
        svg_graph = ""

    past_rows = ""
    for s in reversed(history.data["sessions"][-10:]):
        color = "#4ede7b" if s["good_pct"] > 60 else "#ff8c32"
        past_rows += (
            f"<tr><td>{s['date']}</td>"
            f"<td>{s['duration_sec']//60} min</td>"
            f"<td style='color:{color}'>{s['good_pct']}%</td>"
            f"<td>{s['avg_score']}</td>"
            f"<td>{s['best_score']}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>PostureGhost - Rapport de session</title>
<style>
  body {{ font-family: sans-serif; background: #0f0c1a; color: #e0daf5; margin: 0; padding: 2rem; }}
  h1 {{ color: #a06cff; }} h2 {{ color: #7dd6f0; margin-top: 2rem; }}
  .cards {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .card {{ background: #1c1630; border-radius: 12px; padding: 1.2rem 1.8rem; min-width: 160px; }}
  .card .val {{ font-size: 2rem; font-weight: bold; }}
  .card .lbl {{ font-size: 0.8rem; color: #9090b0; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #2a2440; }}
  th {{ color: #a06cff; }}
</style>
</head>
<body>
<h1>PostureGhost v3.0 - Rapport de session</h1>
<p style="color:#7090a0">{datetime.now().strftime('%A %d %B %Y, %H:%M')}</p>
<div class="cards">
  <div class="card"><div class="val" style="color:#4ede7b">{good_pct:.0f}%</div><div class="lbl">Bonne posture</div></div>
  <div class="card"><div class="val" style="color:#a06cff">{avg_score:.0f}</div><div class="lbl">Score moyen</div></div>
  <div class="card"><div class="val" style="color:#7dd6f0">{best_score}</div><div class="lbl">Meilleur score</div></div>
  <div class="card"><div class="val" style="color:#ffcc55">{int(duration_sec)//60} min</div><div class="lbl">Duree</div></div>
  <div class="card"><div class="val" style="color:#ff8c32">{history.best_streak()}</div><div class="lbl">Streak sessions</div></div>
  <div class="card"><div class="val" style="color:#c0c0e0">{history.all_time_avg()}</div><div class="lbl">Moy. historique</div></div>
</div>
<h2>Courbe de score</h2>
{svg_graph}
<h2>10 dernieres sessions</h2>
<table>
  <tr><th>Date</th><th>Duree</th><th>Bonne posture</th><th>Score moy.</th><th>Meilleur</th></tr>
  {past_rows}
</table>
<p style="color:#3a3460;margin-top:3rem">Genere par PostureGhost v3.0</p>
</body>
</html>"""

    with open(fn, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Rapport HTML : {fn}")
    return fn

# ============================================================
# HELPERS DESSIN
# ============================================================

def rounded_rect(img, x1, y1, x2, y2, color, r=12, fill=True, alpha=1.0):
    if img is None:
        return
    x1, y1, x2, y2, r = int(x1), int(y1), int(x2), int(y2), max(1, int(r))
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
    r = max(1, h//2)
    rounded_rect(img, x, y, x+w, y+h, color_bg, r=r)
    fill = min(w, max(h, int(pct/100*w)))
    rounded_rect(img, x, y, x+fill, y+h, color_fill, r=r)


def draw_arc_ring(img, cx, cy, radius, thickness, pct, color, bg_color=(55, 50, 75)):
    cv2.ellipse(img, (int(cx), int(cy)), (int(radius), int(radius)),
                0, 0, 360, bg_color, int(thickness), cv2.LINE_AA)
    if pct > 0:
        cv2.ellipse(img, (int(cx), int(cy)), (int(radius), int(radius)),
                    0, -90, -90 + int(360 * pct / 100), color, int(thickness), cv2.LINE_AA)


def draw_score_ring(frame, cx, cy, score):
    col = score_color(score)
    draw_arc_ring(frame, cx, cy, 56, 10, score, col)
    if score < CRITICAL_THRESHOLD and int(time.time() * 4) % 2:
        draw_arc_ring(frame, cx, cy, 64, 3, 100, COLOR_RED)
    s_str = str(score)
    tw = len(s_str) * 14
    cv2.putText(frame, s_str, (cx - tw//2 + 2, cy + 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, col, 2, cv2.LINE_AA)
    cv2.putText(frame, "%", (cx + tw//2 - 2, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def draw_mini_graph(frame, scores, x, y, w, h):
    if len(scores) < 2:
        return
    rounded_rect(frame, x, y, x+w, y+h, (38, 33, 58), r=6, alpha=0.85)
    for thresh, c in [(GOOD_POSTURE_THRESHOLD, COLOR_GREEN),
                      (WARNING_THRESHOLD, COLOR_ORANGE),
                      (CRITICAL_THRESHOLD, COLOR_RED)]:
        gy = y + h - int(thresh / 100 * h)
        cv2.line(frame, (x+2, gy), (x+w-2, gy), c, 1, cv2.LINE_AA)
    n   = len(scores)
    pts = []
    for i, s in enumerate(scores):
        px = x + int(i / max(n-1, 1) * w)
        py = y + h - max(1, int(s / 100 * h))
        pts.append((px, py))
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], score_color(scores[i]), 2, cv2.LINE_AA)


def draw_breathing_guide(frame, w, h, session_time):
    cycle = 8.0
    phase = session_time % cycle
    if phase < 4.0:
        label, progress, col = "INSPIREZ", phase/4.0, (200, 240, 255)
    else:
        label, progress, col = "EXPIREZ",  (phase-4.0)/4.0, (150, 200, 170)
    bx, by = w - 135, h - 95
    draw_arc_ring(frame, bx, by, 22, 5, int(progress*100), col, (40, 40, 60))
    cv2.putText(frame, label,         (bx-28, by+38), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
    cv2.putText(frame, "Respiration", (bx-34, by-30), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (110,110,140), 1, cv2.LINE_AA)


def draw_break_reminder(frame, w, h, time_until_break, total_break_sec):
    if total_break_sec <= 0:
        return
    pct    = min(100, (total_break_sec - time_until_break) / total_break_sec * 100)
    bx, by = 340, h - 46
    bar_w  = 280
    rounded_rect(frame, bx, by, bx+bar_w, by+13, (38,33,58), r=6, alpha=0.85)
    col = COLOR_CYAN if pct < 70 else (COLOR_ORANGE if pct < 90 else COLOR_RED)
    rounded_rect(frame, bx, by, bx+max(6,int(pct/100*bar_w)), by+13, col, r=6)
    mins, secs = int(time_until_break//60), int(time_until_break%60)
    cv2.putText(frame, f"Pause dans {mins:02d}:{secs:02d}",
                (bx+6, by+10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_WHITE, 1, cv2.LINE_AA)
    if time_until_break <= 0:
        tx = w//2 - 110
        rounded_rect(frame, tx, h//2-30, tx+220, h//2+30, (50,140,80), r=12, alpha=0.92)
        cv2.putText(frame, "Pause ! Levez-vous 5 min",
                    (tx+8, h//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_stretch_reminder(frame, w, h, exercise_text):
    tw = len(exercise_text) * 9
    tx = w//2 - tw//2
    rounded_rect(frame, tx-14, h//2+50, tx+tw+14, h//2+86, (30,80,120), r=12, alpha=0.92)
    cv2.putText(frame, f"Stretch : {exercise_text}",
                (tx, h//2+74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,230,255), 1, cv2.LINE_AA)


def draw_trend_indicator(frame, scores, x, y):
    if len(scores) < 20:
        return
    delta = scores[-1] - scores[-20]
    if delta > 5:
        symbol, col = "^", COLOR_GREEN
    elif delta < -5:
        symbol, col = "v", COLOR_RED
    else:
        symbol, col = "~", (130, 130, 160)
    cv2.putText(frame, symbol,           (x,    y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{delta:+.0f}", (x+18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)


def draw_tip_panel(frame, w, h, tip_text):
    bx, by = 16, h - 58
    tw = len(tip_text) * 8 + 20
    rounded_rect(frame, bx, by, bx+tw, by+26, (28,22,50), r=8, alpha=0.82)
    cv2.putText(frame, f"Conseil : {tip_text}",
                (bx+8, by+17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,160,255), 1, cv2.LINE_AA)


def draw_distraction_warning(frame, w, h):
    rounded_rect(frame, w//2-160, h-100, w//2+160, h-68, (50,30,80), r=10, alpha=0.88)
    cv2.putText(frame, "Regardez votre ecran !",
                (w//2-130, h-78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_ACCENT, 1, cv2.LINE_AA)


def draw_shrug_warning(frame, w):
    rounded_rect(frame, w//2-180, 58, w//2+180, 92, (60,35,20), r=10, alpha=0.88)
    cv2.putText(frame, "Detendez vos epaules !",
                (w//2-148, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_ORANGE, 1, cv2.LINE_AA)


def draw_fatigue_warning(frame, w, h):
    rounded_rect(frame, w-260, h-100, w-10, h-68, (40,20,60), r=10, alpha=0.88)
    cv2.putText(frame, "Fatigue detectee !",
                (w-248, h-78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,140,255), 1, cv2.LINE_AA)


def draw_neck_angle(frame, landmarks, frame_shape, neck_angle):
    h, w = frame_shape[:2]
    ls   = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    sx   = int(ls.x * w) - 50
    sy   = int(ls.y * h) - 15
    col  = COLOR_GREEN if neck_angle < 20 else (COLOR_ORANGE if neck_angle < 40 else COLOR_RED)
    cv2.putText(frame, f"Cou:{neck_angle:.0f}d",
                (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)


def draw_session_timer(frame, elapsed_sec, x, y):
    m, s = int(elapsed_sec//60), int(elapsed_sec%60)
    cv2.putText(frame, f"{m:02d}:{s:02d}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,160), 1, cv2.LINE_AA)

# ============================================================
# SKELETONS
# ============================================================

def _joint_strain(lm_idx, back_angle, head_fwd, shoulder_tilt):
    back_bad = min(1.0, back_angle / 55.0)
    head_bad = min(1.0, abs(head_fwd) / 110.0)
    sh_bad   = min(1.0, shoulder_tilt / 50.0)
    spine_j  = {mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value}
    neck_j   = {mp_pose.PoseLandmark.LEFT_EAR.value,
                mp_pose.PoseLandmark.RIGHT_EAR.value,
                mp_pose.PoseLandmark.NOSE.value}
    if lm_idx in spine_j:
        return max(back_bad, sh_bad)
    if lm_idx in neck_j:
        return head_bad
    return 0.0


def draw_user_skeleton(frame, landmarks, back_angle=0, head_fwd=0, shoulder_tilt=0):
    if frame is None:
        return
    h, w = frame.shape[:2]
    for (si, ei) in mp_pose.POSE_CONNECTIONS:
        strain = (_joint_strain(si, back_angle, head_fwd, shoulder_tilt) +
                  _joint_strain(ei, back_angle, head_fwd, shoulder_tilt)) / 2
        col = (int(strain*180), int(220 - strain*190), int(100 - strain*100))
        cv2.line(frame,
                 (int(landmarks[si].x*w), int(landmarks[si].y*h)),
                 (int(landmarks[ei].x*w), int(landmarks[ei].y*h)),
                 col, 2, cv2.LINE_AA)
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x*w), int(lm.y*h)
        strain = _joint_strain(idx, back_angle, head_fwd, shoulder_tilt)
        col = (int(strain*200), int(220 - strain*200), int(100 - strain*100))
        cv2.circle(frame, (cx, cy), 5, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)


def draw_ghost_skeleton(frame, landmarks, calibrated_landmarks, frame_shape, posture_score):
    if frame is None:
        return frame
    h, w, _ = frame_shape
    ox      = int(w * 0.52)
    source  = calibrated_landmarks if calibrated_landmarks else landmarks
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
    lx          = int(w * 0.73)
    ghost_label = "CALIBRE" if calibrated_landmarks else "OBJECTIF"
    ghost_sub   = "Ta posture ideale" if calibrated_landmarks else "Posture ideale"
    cv2.putText(frame, ghost_label, (lx-38, 50), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_GHOST, 1, cv2.LINE_AA)
    cv2.putText(frame, ghost_sub,   (lx-50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,180,255), 1, cv2.LINE_AA)
    if posture_score < WARNING_THRESHOLD:
        ax, ay = int(w*0.61), int(h*0.45)
        cv2.arrowedLine(frame, (ax, ay), (ax+80, ay), COLOR_CYAN, 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(frame, "Copiez-moi !", (ax-10, ay-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)
    return frame

# ============================================================
# PANNEAUX HUD
# ============================================================

def draw_left_panel(frame, score, back_angle, head_fwd, shoulder_tilt,
                    fps, good_pct, muted, scores_history, sitting,
                    avg_score, session_elapsed):
    pw, ph = 330, 530
    px, py = 16, 16
    rounded_rect(frame, px, py, px+pw, py+ph, panel_bg(), r=18, alpha=0.91)

    night_label = " [Nuit]" if night_mode_active() else ""
    cv2.putText(frame, f"POSTUREGHOST{night_label}", (px+16, py+30),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, COLOR_ACCENT, 1, cv2.LINE_AA)

    pos_col = (130,200,255) if sitting else (100,230,130)
    rounded_rect(frame, px+200, py+14, px+310, py+34, (45,38,70), r=8)
    cv2.putText(frame, "Assis" if sitting else "Debout",
                (px+212, py+28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pos_col, 1, cv2.LINE_AA)

    cv2.line(frame, (px+16, py+42), (px+pw-16, py+42), (55,48,80), 1)

    ring_cx, ring_cy = px+74, py+120
    draw_score_ring(frame, ring_cx, ring_cy, score)
    status, s_color = get_status(score)
    cv2.putText(frame, status, (ring_cx-28, ring_cy+75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, s_color, 1, cv2.LINE_AA)

    draw_trend_indicator(frame, list(scores_history), px+162, ring_cy-10)
    cv2.putText(frame, f"Moy:{avg_score:.0f}", (px+162, ring_cy+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,200), 1, cv2.LINE_AA)
    draw_session_timer(frame, session_elapsed, px+162, ring_cy+44)

    draw_mini_graph(frame, list(scores_history), px+148, ring_cy-55, 162, 100)

    cv2.line(frame, (px+16, py+208), (px+pw-16, py+208), (55,48,80), 1)

    def metric(label, val_str, badness, y0):
        bc = COLOR_GREEN if badness < 30 else (COLOR_ORANGE if badness < 65 else COLOR_RED)
        cv2.putText(frame, label,   (px+16, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (175,175,210), 1, cv2.LINE_AA)
        cv2.putText(frame, val_str, (px+230, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
        progress_bar(frame, px+16, y0+5, 298, 7, min(100, badness), bc)

    metric("Inclinaison dos",   f"{back_angle:.1f}d",     min(100, back_angle*1.8),    py+228)
    metric("Tete en avant",     f"{abs(head_fwd):.0f}px", min(100, abs(head_fwd)*0.9), py+268)
    metric("Asymetrie epaules", f"{shoulder_tilt:.0f}px", min(100, shoulder_tilt*2.0), py+308)

    cv2.line(frame, (px+16, py+336), (px+pw-16, py+336), (55,48,80), 1)

    cv2.putText(frame, "Bonne posture session", (px+16, py+358),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150,150,190), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{good_pct:.0f}%", (px+242, py+358),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLOR_CYAN, 1, cv2.LINE_AA)
    progress_bar(frame, px+16, py+366, 298, 10, good_pct, COLOR_CYAN)

    good_sec = int(good_pct / 100 * session_elapsed)
    bad_sec  = int(session_elapsed - good_sec)
    cv2.putText(frame, f"Bonne: {good_sec//60}m{good_sec%60:02d}s",
                (px+16, py+394), cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Mauvaise: {bad_sec//60}m{bad_sec%60:02d}s",
                (px+162, py+394), cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_RED, 1, cv2.LINE_AA)

    cv2.line(frame, (px+16, py+410), (px+pw-16, py+410), (55,48,80), 1)

    mute_color = COLOR_RED if muted else COLOR_GREEN
    cv2.circle(frame, (px+26, py+432), 7, mute_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "Son COUPE" if muted else "Son ACTIF",
                (px+40, py+437), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mute_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:.0f} FPS", (px+250, py+ph-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,110), 1, cv2.LINE_AA)


def draw_top_alert(frame, score):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if score < WARNING_THRESHOLD:
        color = COLOR_RED if score < CRITICAL_THRESHOLD else COLOR_ORANGE
        msg   = ("DANGER ! Redressez-vous IMMEDIATEMENT"
                 if score < CRITICAL_THRESHOLD
                 else "Mauvaise posture - Suivez le fantome")
        rounded_rect(frame, w//2-300, 10, w//2+300, 52, color, r=10, alpha=0.78)
        cv2.putText(frame, msg, (w//2-240, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, COLOR_WHITE, 1, cv2.LINE_AA)
    if score < CRITICAL_THRESHOLD and int(time.time()*3) % 2:
        cv2.rectangle(frame, (4,4), (w-4,h-4), COLOR_RED, 3)
    return frame


def draw_ghost_zone_bg(frame):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    rounded_rect(frame, int(w*0.50), 8, w-8, h-8, (20,16,36), r=14, alpha=0.45)
    return frame


def draw_calibration_overlay(frame, countdown):
    h, w = frame.shape[:2]
    rounded_rect(frame, w//2-240, h//2-55, w//2+240, h//2+55, (30,25,50), r=16, alpha=0.92)
    cv2.putText(frame, "CALIBRATION",
                (w//2-90, h//2-20), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Adoptez votre meilleure posture... {countdown}",
                (w//2-200, h//2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    draw_arc_ring(frame, w//2+185, h//2, 20, 4, int((3-countdown)/3*100), COLOR_ACCENT)


def draw_help_overlay(frame):
    h, w = frame.shape[:2]
    rounded_rect(frame, w//2-290, h//2-195, w//2+290, h//2+195, (18,14,36), r=18, alpha=0.96)
    cv2.putText(frame, "RACCOURCIS CLAVIER",
                (w//2-130, h//2-165), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_ACCENT, 1, cv2.LINE_AA)
    shortcuts = [
        ("Q", "Quitter l'application"),
        ("R", "Reinitialiser les stats"),
        ("S", "Capturer une screenshot"),
        ("M", "Mute / Activer le son"),
        ("C", "Calibrer le fantome"),
        ("B", "Toggle respiration guidee"),
        ("P", "Toggle rappel de pause"),
        ("H", "Afficher / cacher cette aide"),
    ]
    for i, (key, desc) in enumerate(shortcuts):
        yy = h//2 - 120 + i * 38
        rounded_rect(frame, w//2-270, yy-16, w//2-220, yy+8, (50,40,90), r=6)
        cv2.putText(frame, key,  (w//2-257, yy+2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (w//2-205, yy+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,195,230), 1, cv2.LINE_AA)
    cv2.putText(frame, "Appuyez sur H pour fermer",
                (w//2-120, h//2+170), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,140), 1, cv2.LINE_AA)


def draw_motivation(frame, last_scores):
    if frame is None or len(last_scores) < 10:
        return
    h, w  = frame.shape[:2]
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
    bx = w - tw - 165
    by = h - 58
    rounded_rect(frame, bx-10, by-20, bx+tw+10, by+12, panel_bg(), r=8, alpha=0.82)
    cv2.putText(frame, txt, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)


def draw_gamification_panel(frame, gm, new_badge=None):
    h, w     = frame.shape[:2]
    summ     = gm.summary()
    bar_x, bar_y, bar_w = 340, h-28, 300
    pct      = min(1.0, summ["xp"] / max(1, summ["next_xp"]))
    rounded_rect(frame, bar_x, bar_y, bar_x+bar_w, bar_y+16, (38,32,65), r=8, alpha=0.85)
    rounded_rect(frame, bar_x, bar_y, bar_x+max(16,int(pct*bar_w)), bar_y+16, (130,100,220), r=8)
    cv2.putText(frame, f"Niv. {summ['level']}  {summ['xp']} XP",
                (bar_x+8, bar_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)
    sx = bar_x + bar_w + 12
    cv2.putText(frame, f"Serie: {summ['streak']}j", (sx, bar_y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                COLOR_CYAN if summ["streak"] >= 3 else (130,130,160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{summ['badges']} badges", (sx+90, bar_y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,140,210), 1, cv2.LINE_AA)
    if new_badge:
        msg = f"{new_badge['icon']} {new_badge['name']}  +{new_badge['xp']} XP"
        tw  = len(msg) * 10
        tx  = w//2 - tw//2
        rounded_rect(frame, tx-12, 62, tx+tw+12, 94, (80,60,160), r=12, alpha=0.92)
        cv2.putText(frame, msg, (tx, 84),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_controls(frame):
    if frame is None:
        return
    h = frame.shape[0]
    cv2.putText(frame,
                "Q:Quit  R:Reset  S:Cap  M:Mute  C:Calib  B:Resp  P:Pause  H:Aide",
                (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80,80,115), 1, cv2.LINE_AA)

# ============================================================
# SESSION LOGGER
# ============================================================

class SessionLogger:
    def __init__(self):
        self.rows       = []
        self.start_time = time.time()

    def log(self, score, back, head, shoulder, neck, sitting, shrug, distracted):
        self.rows.append({
            "t_sec":         round(time.time() - self.start_time, 1),
            "score":         score,
            "back_angle":    round(back, 2),
            "head_fwd":      round(head, 2),
            "shoulder_tilt": round(shoulder, 2),
            "neck_angle":    round(neck, 2),
            "sitting":       int(sitting),
            "shrug":         int(shrug),
            "distracted":    int(distracted),
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

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  POSTUREGHOST v3.0  -  Coach IA + Voix + XP + Rapports")
    print("=" * 65)
    print("  Q:Quit  R:Reset  S:Screenshot  M:Mute  C:Calibrer")
    print("  B:Respiration  P:Pause ON/OFF  H:Aide")
    print("=" * 65)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la webcam.")
        return

    os.makedirs("screenshots", exist_ok=True)

    voice   = VoiceCoach(lang='fr')
    history = SessionHistory()

    if GAMIFICATION_AVAILABLE:
        gm = GamificationManager()
        gm.start_session()
        print(f"  Niveau : {gm.summary()['level']}  |  XP : {gm.stats['xp']}")
    else:
        gm = None

    logger = SessionLogger() if EXPORT_CSV else None

    # ---- Etat ----
    frame_count       = 0
    fps               = 30.0
    prev_time         = time.time()
    session_start     = time.time()
    scores_history    = deque(maxlen=90)
    all_scores        = []
    good_posture_time = 0.0
    total_time        = 0.0

    last_score    = 0
    last_back     = 0.0
    last_head_fwd = 0.0
    last_shoulder = 0.0
    last_neck     = 0.0
    last_landmarks  = None
    sitting         = False
    shrug           = False
    distracted      = False
    fatigued        = False

    calibrated_landmarks = None
    calibrating          = False
    calib_start          = 0.0
    CALIB_DURATION       = 3.0

    active_badge     = None
    badge_show_until = 0.0

    break_timer_active = BREAK_REMINDER_SEC > 0
    time_of_last_break = time.time()

    stretch_active     = True
    time_last_stretch  = time.time()
    stretch_idx        = 0
    show_stretch_until = 0.0

    show_breathing = True
    show_help      = False

    tip_idx       = 0
    last_tip_time = time.time()
    TIP_INTERVAL  = 15.0

    print("=" * 65)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Erreur lecture webcam.")
            break

        frame = cv2.flip(frame, 1)
        now          = time.time()
        elapsed      = now - prev_time
        fps          = 0.9*fps + 0.1/elapsed if elapsed > 0 else fps
        prev_time    = now
        session_time = now - session_start
        frame_count += 1

        # Rotation des conseils
        if now - last_tip_time > TIP_INTERVAL:
            tip_idx       = (tip_idx + 1) % len(POSTURE_TIPS)
            last_tip_time = now

        # ---- Traitement pose ----
        if frame_count % FRAME_SKIP == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            frame   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if frame is None:
                continue

            if results.pose_landmarks:
                last_landmarks = results.pose_landmarks.landmark
                (last_score, last_back, last_head_fwd,
                 last_shoulder, last_neck) = calculate_posture_score(
                    last_landmarks, frame.shape)

                sitting    = detect_sitting(last_landmarks, frame.shape)
                shrug      = detect_shrug(last_landmarks, frame.shape)
                distracted = detect_distraction(last_landmarks, frame.shape)

                scores_history.append(last_score)
                all_scores.append(last_score)

                dt = 1.0/fps if fps > 0 else 0
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
                        voice.say(f"Badge debloque : {earned[0]['name']} !", force=True)

                fatigued = detect_fatigue(scores_history)

                alert, force = get_voice_alert(
                    last_score, last_back, last_head_fwd, last_shoulder)
                if alert:
                    voice.say(alert, force=force)
                if shrug:
                    voice.say("Detendez vos epaules.")
                if distracted:
                    voice.say("Regardez votre ecran.")
                if fatigued:
                    voice.say("Vous semblez fatigue, faites une pause.", force=True)

                if logger and frame_count % 15 == 0:
                    logger.log(last_score, last_back, last_head_fwd,
                               last_shoulder, last_neck, sitting, shrug, distracted)

                if calibrating and (now - calib_start) >= CALIB_DURATION:
                    calibrated_landmarks = list(last_landmarks)
                    calibrating = False
                    voice.say("Calibration enregistree !", force=True)
                    print("  Calibration OK.")

        # Stretch reminder
        if stretch_active and (now - time_last_stretch) >= STRETCH_INTERVAL:
            show_stretch_until = now + 5.0
            time_last_stretch  = now
            ex = STRETCH_EXERCISES[stretch_idx % len(STRETCH_EXERCISES)]
            stretch_idx += 1
            voice.say(f"Exercice : {ex}", force=True)

        if active_badge and now > badge_show_until:
            active_badge = None

        # Pause reminder
        time_until_break = 0
        if break_timer_active and BREAK_REMINDER_SEC > 0:
            time_until_break = max(0, BREAK_REMINDER_SEC - (now - time_of_last_break))
            if time_until_break == 0 and now - time_of_last_break > BREAK_REMINDER_SEC:
                voice.say("C'est l'heure de la pause ! Levez-vous.", force=True)
                time_of_last_break = now

        # ---- Dessin ----
        if last_landmarks is not None:
            draw_user_skeleton(frame, last_landmarks, last_back, last_head_fwd, last_shoulder)
            frame = draw_ghost_zone_bg(frame)
            frame = draw_ghost_skeleton(frame, last_landmarks, calibrated_landmarks,
                                        frame.shape, last_score)
            draw_neck_angle(frame, last_landmarks, frame.shape, last_neck)
        else:
            hf, wf = frame.shape[:2]
            rounded_rect(frame, wf//2-220, hf//2-35, wf//2+220, hf//2+35,
                         panel_bg(), r=12, alpha=0.85)
            cv2.putText(frame, "Placez-vous devant la webcam",
                        (wf//2-195, hf//2+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)

        good_pct  = (good_posture_time / total_time * 100) if total_time > 0 else 0
        avg_score = float(np.mean(all_scores)) if all_scores else 0.0
        hf, wf    = frame.shape[:2]

        draw_left_panel(frame, last_score, last_back, last_head_fwd,
                        last_shoulder, fps, good_pct, voice.muted,
                        scores_history, sitting, avg_score, session_time)
        frame = draw_top_alert(frame, last_score)
        draw_motivation(frame, list(scores_history))

        if gm:
            draw_gamification_panel(frame, gm, active_badge)

        if break_timer_active and BREAK_REMINDER_SEC > 0:
            draw_break_reminder(frame, wf, hf, time_until_break, BREAK_REMINDER_SEC)

        if show_breathing:
            draw_breathing_guide(frame, wf, hf, session_time)

        if now < show_stretch_until and last_landmarks is not None:
            draw_stretch_reminder(frame, wf, hf,
                                  STRETCH_EXERCISES[(stretch_idx-1) % len(STRETCH_EXERCISES)])

        if shrug:
            draw_shrug_warning(frame, wf)
        if distracted:
            draw_distraction_warning(frame, wf, hf)
        if fatigued:
            draw_fatigue_warning(frame, wf, hf)

        draw_tip_panel(frame, wf, hf, POSTURE_TIPS[tip_idx])

        if calibrating:
            countdown = max(0, int(CALIB_DURATION - (now - calib_start)) + 1)
            draw_calibration_overlay(frame, countdown)

        if show_help:
            draw_help_overlay(frame)

        draw_controls(frame)

        cv2.imshow("PostureGhost v3.0", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            good_posture_time = 0.0
            total_time        = 0.0
            scores_history.clear()
            all_scores.clear()
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
            if last_landmarks is not None:
                calibrating = True
                calib_start = time.time()
                print("  Calibration demarree...")
            else:
                print("  Aucun squelette detecte.")
        elif key == ord('b'):
            show_breathing = not show_breathing
        elif key == ord('p'):
            break_timer_active = not break_timer_active
            print(f"Rappel pause : {'ON' if break_timer_active else 'OFF'}")
        elif key == ord('h'):
            show_help = not show_help

    # ---- Cleanup ----
    if gm:
        gm.end_session(good_posture_time, last_score)
    voice.stop()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    if logger:
        logger.save()

    avg_score  = float(np.mean(all_scores)) if all_scores else 0.0
    best_score = int(max(all_scores)) if all_scores else 0
    good_pct   = (good_posture_time / total_time * 100) if total_time > 0 else 0

    history.save_session(total_time, good_pct, avg_score, best_score)

    if EXPORT_HTML_REPORT and all_scores:
        generate_html_report(total_time, good_pct, avg_score, best_score,
                             all_scores, history)

    print("\n" + "=" * 65)
    print("  SESSION TERMINEE")
    print("=" * 65)
    if total_time > 0:
        print(f"  Duree           : {total_time:.0f} sec ({int(total_time)//60} min)")
        print(f"  Bonne posture   : {good_posture_time:.0f} sec  ({good_pct:.1f}%)")
        print(f"  Score moyen     : {avg_score:.1f}")
        print(f"  Meilleur score  : {best_score}")
        print(f"  Streak sessions : {history.best_streak()}")
        print(f"  Moy. historique : {history.all_time_avg()}")
    if gm:
        print(f"  XP cette session: +{gm.xp_gained}")
        print(f"  Total XP        : {gm.stats['xp']}")
        print(f"  Niveau          : {gm.summary()['level']}")
        print(f"  Badges          : {gm.stats['unlocked']}")
    print("=" * 65)


if __name__ == "__main__":
    main()