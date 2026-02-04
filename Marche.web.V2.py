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
# MEDIAPIPE
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
# NORMES
# ==============================
def norm_curve(joint, n):
    x = np.linspace(0, 100, n)
    if joint == "Genou":
        return np.interp(x, [0,15,40,60,80,100], [5,15,5,40,60,5])
    if joint == "Hanche":
        return np.interp(x, [0,30,60,100], [30,0,-10,30])
    if joint == "Cheville":
        return np.interp(x, [0,10,50,70,100], [5,10,25,10,5])
    return np.zeros(n)

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
    for side, suf in [("LEFT","G"),("RIGHT","D")]:
        kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(L,f"{side}_SHOULDER"))
        kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(L,f"{side}_HIP"))
        kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(L,f"{side}_KNEE"))
        kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(L,f"{side}_ANKLE"))
        kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(L,f"{side}_HEEL"))
        kp[f"Orteil {suf}"], kp[f"Orteil {suf} vis"] = pt(getattr(L,f"{side}_FOOT_INDEX"))
    return kp

# ==============================
# ANGLES
# ==============================
def angle(a,b,c):
    ba = a-b; bc = c-b
    ba[1]*=-1; bc[1]*=-1
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def angle_hanche(e,h,g): return 180-angle(e,h,g)
def angle_genou(h,g,c): return 180-angle(h,g,c)
def angle_cheville(g,c,o): return angle(g,c,o)

# ==============================
# BANDPASS
# ==============================
def bandpass(sig,lvl,fs=FPS):
    low = 0.3+lvl*0.02
    high = max(6.0-lvl*0.25,low+0.4)
    b,a = butter(2,[low/(fs/2),high/(fs/2)],btype="band")
    return filtfilt(b,a,sig)

# ==============================
# CYCLE
# ==============================
def detect_cycle(y):
    y = np.array(y,dtype=float)
    if np.isnan(y).any():
        idx=np.arange(len(y)); ok=~np.isnan(y)
        if ok.sum()>=2: y=np.interp(idx,idx[ok],y[ok])
        else: return None
    inv=-y
    p,_=find_peaks(inv,distance=FPS//2,prominence=np.std(inv)*0.3)
    return (p[0],p[1]) if len(p)>=2 else None

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path,conf):
    cap=cv2.VideoCapture(path)
    res={k:[] for k in ["Hanche G","Hanche D","Genou G","Genou D","Cheville G","Cheville D"]}
    heelG,heelD=[],[]
    frames=[]
    while cap.isOpened():
        r,f=cap.read()
        if not r: break
        frames.append(f.copy())
        kp=detect_pose(f)
        if kp is None:
            for k in res: res[k].append(np.nan)
            heelG.append(np.nan); heelD.append(np.nan); continue
        def ok(n): return kp[f"{n} vis"]>=conf
        res["Hanche G"].append(angle_hanche(kp["Epaule G"],kp["Hanche G"],kp["Genou G"]) if ok("Epaule G") else np.nan)
        res["Hanche D"].append(angle_hanche(kp["Epaule D"],kp["Hanche D"],kp["Genou D"]) if ok("Epaule D") else np.nan)
        res["Genou G"].append(angle_genou(kp["Hanche G"],kp["Genou G"],kp["Cheville G"]) if ok("Genou G") else np.nan)
        res["Genou D"].append(angle_genou(kp["Hanche D"],kp["Genou D"],kp["Cheville D"]) if ok("Genou D") else np.nan)
        res["Cheville G"].append(angle_cheville(kp["Genou G"],kp["Cheville G"],kp["Orteil G"]) if ok("Orteil G") else np.nan)
        res["Cheville D"].append(angle_cheville(kp["Genou D"],kp["Cheville D"],kp["Orteil D"]) if ok("Orteil D") else np.nan)
        heelG.append(kp["Talon G"][1] if ok("Talon G") else np.nan)
        heelD.append(kp["Talon D"][1] if ok("Talon D") else np.nan)
    cap.release()
    return res,heelG,heelD,frames

# ==============================
# UI
# ==============================
with st.sidebar:
    nom=st.text_input("Nom","DURAND")
    prenom=st.text_input("Prénom","Jean")
    camera_pos=st.selectbox("Angle de film",["Devant","Droite","Gauche"])
    phase_cote=st.selectbox("Phases",["Aucune","Droite","Gauche","Les deux"])
    smooth=st.slider("Lissage",0,10,3)
    conf=st.slider("Seuil confiance",0.1,0.9,0.3,0.05)

video=st.file_uploader("Vidéo",["mp4","avi","mov"])

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp=tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read()); tmp.close()

    data,heelG,heelD,frames=process_video(tmp.name,conf)
    os.unlink(tmp.name)

    phases=[]
    if phase_cote in ["Gauche","Les deux"]:
        c=detect_cycle(heelG)
        if c: phases.append((*c,"orange"))
    if phase_cote in ["Droite","Les deux"]:
        c=detect_cycle(heelD)
        if c: phases.append((*c,"blue"))

    for joint in ["Hanche","Genou","Cheville"]:
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4),gridspec_kw={"width_ratios":[2,1]})
        g=bandpass(np.nan_to_num(data[f"{joint} G"]),smooth)
        d=bandpass(np.nan_to_num(data[f"{joint} D"]),smooth)
        ax1.plot(g,label="Gauche",color="red")
        ax1.plot(d,label="Droite",color="blue")
        for c0,c1,col in phases:
            ax1.axvspan(c0,c1,color=col,alpha=0.3)
        ax1.set_title(joint); ax1.legend()
        ax2.plot(norm_curve(joint,len(g)),color="green")
        ax2.set_title("Norme")
        st.pyplot(fig)
        plt.close(fig)
