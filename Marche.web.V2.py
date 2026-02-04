# ==============================
# IMPORTS
# ==============================
import streamlit as st
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
import mediapipe as mp

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# ==============================
# CONFIG
# ==============================
st.set_page_config("GaitScan Pro (MediaPipe)", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
FPS = 30

# ==============================
# MEDIAPIPE POSE
# ==============================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

# ==============================
# POSE DETECTION
# ==============================
def detect_pose(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(l):
        p = lm[int(l)]
        return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)

    kp = {}
    for side in ["LEFT", "RIGHT"]:
        suf = "G" if side == "LEFT" else "D"
        kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(L, f"{side}_SHOULDER"))
        kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(L, f"{side}_HIP"))
        kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(L, f"{side}_KNEE"))
        kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(L, f"{side}_ANKLE"))
        kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(L, f"{side}_HEEL"))
        kp[f"Orteil {suf}"], kp[f"Orteil {suf} vis"] = pt(getattr(L, f"{side}_FOOT_INDEX"))

    return kp

# ==============================
# ANGLES
# ==============================
def angle(a, b, c):
    ba = a - b
    bc = c - b
    ba[1] *= -1
    bc[1] *= -1
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_t, -1, 1)))

def angle_hanche(e, h, g): return 180 - angle(e, h, g)
def angle_genou(h, g, c): return 180 - angle(h, g, c)
def angle_cheville(g, c, o): return angle(g, c, o)

# ==============================
# BANDPASS
# ==============================
def bandpass(sig, lvl, fs=FPS):
    low = 0.3 + lvl*0.02
    high = max(6.0 - lvl*0.25, low+0.4)
    b,a = butter(2, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b,a,sig)

# ==============================
# CYCLE DETECTION
# ==============================
def detect_cycle(heel_y):
    y = np.array(heel_y, dtype=float)
    if np.isnan(y).any():
        idx = np.arange(len(y))
        good = ~np.isnan(y)
        if good.sum() >= 2:
            y = np.interp(idx, idx[good], y[good])
        else:
            return None

    inv = -y
    peaks,_ = find_peaks(inv, distance=FPS//2, prominence=np.std(inv)*0.3)
    if len(peaks) >= 2:
        return peaks[0], peaks[1]
    return None

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path, conf):
    cap = cv2.VideoCapture(path)

    res = {k: [] for k in [
        "Hanche G","Hanche D","Genou G","Genou D","Cheville G","Cheville D"
    ]}
    heel_y_G, heel_y_D = [], []
    frames, pose_cache = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame.copy())

        kp = detect_pose(frame)
        pose_cache.append(kp)

        if kp is None:
            for k in res: res[k].append(np.nan)
            heel_y_G.append(np.nan); heel_y_D.append(np.nan)
            continue

        def ok(name): return kp[f"{name} vis"] >= conf

        if ok("Epaule G") and ok("Hanche G") and ok("Genou G"):
            res["Hanche G"].append(angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"]))
        else: res["Hanche G"].append(np.nan)

        if ok("Epaule D") and ok("Hanche D") and ok("Genou D"):
            res["Hanche D"].append(angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"]))
        else: res["Hanche D"].append(np.nan)

        if ok("Hanche G") and ok("Genou G") and ok("Cheville G"):
            res["Genou G"].append(angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"]))
        else: res["Genou G"].append(np.nan)

        if ok("Hanche D") and ok("Genou D") and ok("Cheville D"):
            res["Genou D"].append(angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"]))
        else: res["Genou D"].append(np.nan)

        if ok("Genou G") and ok("Cheville G") and ok("Orteil G"):
            res["Cheville G"].append(angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Orteil G"]))
        else: res["Cheville G"].append(np.nan)

        if ok("Genou D") and ok("Cheville D") and ok("Orteil D"):
            res["Cheville D"].append(angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Orteil D"]))
        else: res["Cheville D"].append(np.nan)

        heel_y_G.append(kp["Talon G"][1] if ok("Talon G") else np.nan)
        heel_y_D.append(kp["Talon D"][1] if ok("Talon D") else np.nan)

    cap.release()
    return res, heel_y_G, heel_y_D, frames, pose_cache

# ==============================
# STREAMLIT UI
# ==============================
with st.sidebar:
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")
    smooth = st.slider("Lissage",0,10,3)
    phase_cote = st.selectbox(
        "Affichage des phases",
        ["Aucune","Droite","Gauche","Les deux"]
    )
    conf = st.slider("Seuil confiance landmarks",0.1,0.9,0.3,0.05)

video = st.file_uploader("Vidéo",["mp4","avi","mov"])

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heel_G, heel_D, frames, _ = process_video(tmp.name, conf)
    os.unlink(tmp.name)

    # phases
    phase_colors = []
    if phase_cote in ["Gauche","Les deux"]:
        c = detect_cycle(heel_G)
        if c: phase_colors.append((*c,"orange"))
    if phase_cote in ["Droite","Les deux"]:
        c = detect_cycle(heel_D)
        if c: phase_colors.append((*c,"blue"))

    for joint in ["Hanche","Genou","Cheville"]:
        fig, ax = plt.subplots(figsize=(10,4))
        g = bandpass(np.nan_to_num(data[f"{joint} G"]), smooth)
        d = bandpass(np.nan_to_num(data[f"{joint} D"]), smooth)

        ax.plot(g,label="Gauche",color="red")
        ax.plot(d,label="Droite",color="blue")

        for c0,c1,col in phase_colors:
            ax.axvspan(c0,c1,color=col,alpha=0.25)

        ax.set_title(joint)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
