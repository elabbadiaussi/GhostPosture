# GhostPosture

GhostPosture est un coach IA de posture qui aide les utilisateurs à améliorer leur ergonomie pendant le travail sur écran. L’application analyse la posture du corps avec la vision par ordinateur, affiche un score en direct et fournit un retour visuel pour encourager de meilleures habitudes.

## Fonctionnalités

- Détection de posture en temps réel avec la webcam
- Score de posture et retour de statut en direct
- Superposition d’un squelette fantôme pour un guide visuel plus clair
- Coach vocal et gamification
- Historique de session, export CSV et rapports HTML

## Modes du projet

Le projet principal est :

- main.py : version desktop, conçue pour une utilisation locale avec OpenCV et la webcam

## Stack technique

- Python
- OpenCV
- MediaPipe
- NumPy
- Streamlit
- pyttsx3

## Structure du dépôt

- main.py - application principale desktop de coaching posture
- gamification.py - XP, séries, badges et statistiques locales
- requirements.txt - dépendances Python
- posture_stats.json - stockage local de la gamification
- sessions/ - rapports de session et exports générés

## Installation locale

### 1. Créer un environnement virtuel

```powershell
python -m venv .venv
```

### 2. L’activer

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Lancer l’application desktop

```powershell
python .\main.py
```

## Notes

- main.py correspond à l’expérience desktop complète et utilise directement la webcam locale.

## Dépendances

Le projet utilise les paquets listés dans requirements.txt, notamment :

- opencv-python
- mediapipe==0.10.14
- numpy
- streamlit
- pillow
- pyttsx3

## Licence

Aucune licence n’a encore été spécifiée.
