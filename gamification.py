"""
POSTUREGHOST — Gamification: XP, streaks, badges
"""

import json
import os
from datetime import date, timedelta

SAVE_FILE = "posture_stats.json"

# ============================================
# BADGE DEFINITIONS
# ============================================

BADGES = [
    {
        "id":    "first_good",
        "name":  "Premier pas",
        "desc":  "Premiere bonne posture detectee",
        "icon":  "★",
        "xp":    50,
        "check": lambda s: s["good_frames"] >= 1,
    },
    {
        "id":    "five_minutes",
        "name":  "5 minutes",
        "desc":  "5 min de bonne posture",
        "icon":  "◆",
        "xp":    75,
        "check": lambda s: s["good_minutes"] >= 5,
    },
    {
        "id":    "streak_3",
        "name":  "Serie x3",
        "desc":  "3 jours de suite",
        "icon":  "▲",
        "xp":    100,
        "check": lambda s: s["streak"] >= 3,
    },
    {
        "id":    "streak_7",
        "name":  "Semaine parfaite",
        "desc":  "7 jours de suite",
        "icon":  "◉",
        "xp":    250,
        "check": lambda s: s["streak"] >= 7,
    },
    {
        "id":    "score_90",
        "name":  "Posture elite",
        "desc":  "Score 90+ atteint",
        "icon":  "✦",
        "xp":    150,
        "check": lambda s: s["best_score"] >= 90,
    },
    {
        "id":    "sessions_10",
        "name":  "Assidu",
        "desc":  "10 sessions completees",
        "icon":  "■",
        "xp":    200,
        "check": lambda s: s["sessions"] >= 10,
    },
    {
        "id":    "perfect_hour",
        "name":  "Heure parfaite",
        "desc":  "60 min de bonne posture",
        "icon":  "●",
        "xp":    300,
        "check": lambda s: s["good_minutes"] >= 60,
    },
]

# ============================================
# LEVEL TABLE
# ============================================

LEVELS = [
    (0,    "Debutant"),
    (100,  "Apprenti"),
    (300,  "Confirme"),
    (600,  "Avance"),
    (1000, "Expert"),
    (1500, "Maitre"),
    (2500, "Legende"),
]

def get_level(xp):
    level, name = LEVELS[0]
    for threshold, lname in LEVELS:
        if xp >= threshold:
            level, name = threshold, lname
    return name

def next_level_xp(xp):
    for i, (threshold, _) in enumerate(LEVELS):
        if xp < threshold:
            return threshold
    return LEVELS[-1][0] + 1000  # past max level

# ============================================
# GAMIFICATION MANAGER
# ============================================

class GamificationManager:
    def __init__(self):
        self.stats = self._load()
        self._update_streak()
        self.new_badges = []   # badges earned this session (for overlay display)
        self.xp_gained  = 0    # XP earned this session

    # ---------- Persistence ----------

    def _default_stats(self):
        return {
            "xp":           0,
            "sessions":     0,
            "streak":       0,
            "last_session": None,
            "best_score":   0,
            "good_frames":  0,
            "good_minutes": 0.0,
            "unlocked":     [],
        }

    def _load(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, "r") as f:
                    data = json.load(f)
                    # fill missing keys for backward compat
                    defaults = self._default_stats()
                    for k, v in defaults.items():
                        data.setdefault(k, v)
                    return data
            except Exception:
                pass
        return self._default_stats()

    def save(self):
        with open(SAVE_FILE, "w") as f:
            json.dump(self.stats, f, indent=2)

    # ---------- Streak ----------

    def _update_streak(self):
        today = str(date.today())
        last  = self.stats.get("last_session")
        if last is None:
            return
        yesterday = str(date.today() - timedelta(days=1))
        if last == yesterday:
            pass          # streak continues, incremented on session start
        elif last != today:
            self.stats["streak"] = 0   # streak broken

    # ---------- Session lifecycle ----------

    def start_session(self):
        today = str(date.today())
        if self.stats["last_session"] != today:
            self.stats["sessions"]     += 1
            self.stats["streak"]       += 1
            self.stats["last_session"]  = today
        self.new_badges = []
        self.xp_gained  = 0

    def end_session(self, good_seconds, best_score):
        """Call this when the user quits. Pass session totals."""
        self.stats["good_minutes"] += good_seconds / 60.0
        if best_score > self.stats["best_score"]:
            self.stats["best_score"] = best_score

        # XP for time in good posture (1 XP per 10 seconds)
        time_xp = int(good_seconds / 10)
        self._add_xp(time_xp)

        self._check_badges()
        self.save()

    # ---------- Real-time updates ----------

    def on_good_frame(self, score):
        """Call every frame where score >= GOOD_POSTURE_THRESHOLD."""
        self.stats["good_frames"] += 1

    def on_score(self, score):
        if score > self.stats["best_score"]:
            self.stats["best_score"] = score
            self._check_badges()

    # ---------- XP & badges ----------

    def _add_xp(self, amount):
        self.stats["xp"] += amount
        self.xp_gained   += amount

    def _check_badges(self):
        for badge in BADGES:
            if badge["id"] not in self.stats["unlocked"]:
                if badge["check"](self.stats):
                    self.stats["unlocked"].append(badge["id"])
                    self._add_xp(badge["xp"])
                    self.new_badges.append(badge)
                    print(f"  [BADGE] {badge['icon']} {badge['name']} +{badge['xp']} XP")

    def pop_new_badges(self):
        """Returns newly earned badges and clears the queue."""
        badges = self.new_badges[:]
        self.new_badges = []
        return badges

    # ---------- Display helpers ----------

    def summary(self):
        s = self.stats
        return {
            "xp":       s["xp"],
            "level":    get_level(s["xp"]),
            "next_xp":  next_level_xp(s["xp"]),
            "streak":   s["streak"],
            "sessions": s["sessions"],
            "badges":   len(s["unlocked"]),
        }