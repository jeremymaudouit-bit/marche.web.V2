# ==============================
# IMPORTS
# ==============================
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
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
st.set_page_config("GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
FPS = 30

# ==============================
# MOVENET
# ==============================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(img[None], 192, 192)
    out = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
    return out["output_0"].numpy()[0, 0]

# ==============================
# JOINTS
# ==============================
J = {
    "Epaule G":5, "Epaule D":6,
    "Hanche G":11, "Hanche D":12,
    "Genou G":13, "Genou D":14,
    "Cheville G":15, "Cheville D":16
}

SEGMENTS = [
    ("Epaule G","Hanche G"), ("Hanche G","Genou G"), ("Genou G","Cheville G"),
    ("Epaule D","Hanche D"), ("Hanche D","Genou D"), ("Genou D","Cheville D"),
    ("Hanche G","Hanche D"), ("Epaule G","Epaule D"),
]

# ==============================
# UTILS
# ==============================
def angle(a,b,c):
    ba, bc = a-b, c-b
    return np.degrees(
        np.arccos(
            np.clip(
                np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6),
                -1,1
            )
        )
    )

def bandpass(sig, level, fs=FPS):
    low = 0.3 + level*0.02
    high = max(6.0 - level*0.25, low+0.4)
    b,a = butter(2, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b,a,sig)

def draw_segments(frame, kp):
    h, w, _ = frame.shape
    for a,b in SEGMENTS:
        ia, ib = J[a], J[b]
        xa,ya,ca = kp[ia]
        xb,yb,cb = kp[ib]
        if ca>0.3 and cb>0.3:
            p1 = (int(xa*w), int(ya*h))
            p2 = (int(xb*w), int(yb*h))
            cv2.line(frame,p1,p2,(0,255,0),2)
            cv2.circle(frame,p1,4,(0,255,0),-1)
            cv2.circle(frame,p2,4,(0,255,0),-1)
    return frame

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path):
    cap = cv2.VideoCapture(path)

    res = {k:[] for k in [
        "Hanche G","Hanche D","Genou G","Genou D",
        "Cheville G","Cheville D"
    ]}
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)
        annotated = draw_segments(frame.copy(), kp)
        frames.append(annotated)

        res["Hanche G"].append(angle(kp[J["Epaule G"],:2], kp[J["Hanche G"],:2], kp[J["Genou G"],:2]))
        res["Hanche D"].append(angle(kp[J["Epaule D"],:2], kp[J["Hanche D"],:2], kp[J["Genou D"],:2]))

        res["Genou G"].append(angle(kp[J["Hanche G"],:2], kp[J["Genou G"],:2], kp[J["Cheville G"],:2]))
        res["Genou D"].append(angle(kp[J["Hanche D"],:2], kp[J["Genou D"],:2], kp[J["Cheville D"],:2]))

        res["Cheville G"].append(angle(kp[J["Genou G"],:2], kp[J["Cheville G"],:2], kp[J["Cheville G"],:2]+[0,1]))
        res["Cheville D"].append(angle(kp[J["Genou D"],:2], kp[J["Cheville D"],:2], kp[J["Cheville D"],:2]+[0,1]))

    cap.release()
    return res, frames

# ==============================
# STREAMLIT UI
# ==============================
with st.sidebar:
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")
    smooth = st.slider("Lissage",0,10,3)

video = st.file_uploader("Vidéo",["mp4","avi","mov"])

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, frames = process_video(tmp.name)
    os.unlink(tmp.name)

    st.subheader("🎥 Vidéo analysée – Segments mesurés")

    idx = st.slider("Frame",0,len(frames)-1,len(frames)//2)
    st.image(frames[idx], channels="BGR", use_container_width=True)

    st.caption(
        f"Hanche D : {data['Hanche D'][idx]:.1f}° | "
        f"Genou D : {data['Genou D'][idx]:.1f}° | "
        f"Cheville D : {data['Cheville D'][idx]:.1f}°"
    )

    st.subheader("📈 Courbes articulaires")

    for joint in ["Hanche","Genou","Cheville"]:
        g = bandpass(np.array(data[f"{joint} G"]), smooth)
        d = bandpass(np.array(data[f"{joint} D"]), smooth)

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(g,label="Gauche",color="red")
        ax.plot(d,label="Droite",color="blue")
        ax.set_title(joint)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
