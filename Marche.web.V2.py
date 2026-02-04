# ==============================
# IMPORTS
# ==============================
import streamlit as st
import cv2, os, tempfile, base64
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
import mediapipe as mp

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

import streamlit.components.v1 as components

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
def angle(a, b, c):
    ba = a - b
    bc = c - b
    ba[1] *= -1
    bc[1] *= -1
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def angle_hanche(e,h,g): return 180 - angle(e,h,g)
def angle_genou(h,g,c): return 180 - angle(h,g,c)
def angle_cheville(g,c,o): return angle(g,c,o)  # tibia–pied (genou-cheville-orteil)

# ==============================
# BANDPASS
# ==============================
def bandpass(sig, lvl, fs=FPS):
    low = 0.3 + lvl*0.02
    high = max(6.0 - lvl*0.25, low+0.4)
    b,a = butter(2, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, sig)

# ==============================
# CYCLE
# ==============================
def detect_cycle(y):
    y = np.array(y, dtype=float)
    if np.isnan(y).any():
        idx = np.arange(len(y)); ok = ~np.isnan(y)
        if ok.sum() >= 2:
            y = np.interp(idx, idx[ok], y[ok])
        else:
            return None
    inv = -y
    p,_ = find_peaks(inv, distance=FPS//2, prominence=np.std(inv)*0.3)
    return (int(p[0]), int(p[1])) if len(p) >= 2 else None

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path, conf):
    cap = cv2.VideoCapture(path)
    res = {k:[] for k in ["Hanche G","Hanche D","Genou G","Genou D","Cheville G","Cheville D"]}
    heelG, heelD = [], []
    frames = []

    while cap.isOpened():
        r, f = cap.read()
        if not r: break
        frames.append(f.copy())

        kp = detect_pose(f)
        if kp is None:
            for k in res: res[k].append(np.nan)
            heelG.append(np.nan); heelD.append(np.nan)
            continue

        def ok(n): return kp.get(f"{n} vis", 0.0) >= conf

        res["Hanche G"].append(
            angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"])
            if (ok("Epaule G") and ok("Hanche G") and ok("Genou G")) else np.nan
        )
        res["Hanche D"].append(
            angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"])
            if (ok("Epaule D") and ok("Hanche D") and ok("Genou D")) else np.nan
        )

        res["Genou G"].append(
            angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"])
            if (ok("Hanche G") and ok("Genou G") and ok("Cheville G")) else np.nan
        )
        res["Genou D"].append(
            angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"])
            if (ok("Hanche D") and ok("Genou D") and ok("Cheville D")) else np.nan
        )

        res["Cheville G"].append(
            angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Orteil G"])
            if (ok("Genou G") and ok("Cheville G") and ok("Orteil G")) else np.nan
        )
        res["Cheville D"].append(
            angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Orteil D"])
            if (ok("Genou D") and ok("Cheville D") and ok("Orteil D")) else np.nan
        )

        heelG.append(float(kp["Talon G"][1]) if ok("Talon G") else np.nan)
        heelD.append(float(kp["Talon D"][1]) if ok("Talon D") else np.nan)

    cap.release()
    return res, heelG, heelD, frames

# ==============================
# ANNOTATION IMAGES (angles + gros)
# ==============================
def draw_angle_on_frame(img_bgr, pA, pB, pC, ang_deg, color=(0,255,0)):
    h, w = img_bgr.shape[:2]
    A = (int(pA[0]*w), int(pA[1]*h))
    B = (int(pB[0]*w), int(pB[1]*h))
    C = (int(pC[0]*w), int(pC[1]*h))

    line_th = 4
    circle_r = 7
    text_scale = 1.2
    text_th = 3

    cv2.line(img_bgr, A, B, color, line_th)
    cv2.line(img_bgr, C, B, color, line_th)
    cv2.circle(img_bgr, B, circle_r, (0,0,255), -1)

    label = f"{int(round(ang_deg))}°"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_th)
    tx, ty = B[0] + 10, B[1] - 10
    cv2.rectangle(img_bgr, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0,0,0), -1)
    cv2.putText(img_bgr, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), text_th, cv2.LINE_AA)

def annotate_frame(frame_bgr, kp, conf=0.30):
    if kp is None:
        return frame_bgr

    def ok(n): return kp.get(f"{n} vis", 0.0) >= conf
    out = frame_bgr.copy()

    if ok("Epaule G") and ok("Hanche G") and ok("Genou G"):
        draw_angle_on_frame(out, kp["Epaule G"], kp["Hanche G"], kp["Genou G"],
                            angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"]))
    if ok("Epaule D") and ok("Hanche D") and ok("Genou D"):
        draw_angle_on_frame(out, kp["Epaule D"], kp["Hanche D"], kp["Genou D"],
                            angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"]))

    if ok("Hanche G") and ok("Genou G") and ok("Cheville G"):
        draw_angle_on_frame(out, kp["Hanche G"], kp["Genou G"], kp["Cheville G"],
                            angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"]))
    if ok("Hanche D") and ok("Genou D") and ok("Cheville D"):
        draw_angle_on_frame(out, kp["Hanche D"], kp["Genou D"], kp["Cheville D"],
                            angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"]))

    if ok("Genou G") and ok("Cheville G") and ok("Orteil G"):
        draw_angle_on_frame(out, kp["Genou G"], kp["Cheville G"], kp["Orteil G"],
                            angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Orteil G"]))
    if ok("Genou D") and ok("Cheville D") and ok("Orteil D"):
        draw_angle_on_frame(out, kp["Genou D"], kp["Cheville D"], kp["Orteil D"],
                            angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Orteil D"]))

    return out

# ==============================
# STEP LENGTH + ASYMMETRY
# ==============================
def nan_interp(x):
    x = np.array(x, dtype=float)
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], x[ok])
    return None

def asym_percent(left, right):
    """
    Asymétrie (%) = 100 * |R - L| / ((R + L)/2)
    Retourne None si impossible.
    """
    if left is None or right is None:
        return None
    denom = (left + right) / 2.0
    if abs(denom) < 1e-6:
        return None
    return 100.0 * abs(right - left) / abs(denom)

def compute_step_length_cm(heelG, heelD, taille_cm):
    """
    Estime une longueur de pas en cm à partir des signaux y (talons) + taille.
    Renvoie:
      - step_mean_cm, step_std_cm
      - step_G_cm (pas "côté gauche"), step_D_cm (pas "côté droit")
      - step_asym_pct
    Note: estimation monocaméra 2D sans calibration.
    """
    hG = nan_interp(heelG)
    hD = nan_interp(heelD)
    if hG is None or hD is None:
        return None, None, None, None, None

    cG = detect_cycle(hG)
    cD = detect_cycle(hD)

    # On fabrique deux estimations simples (une depuis cycle G, une depuis cycle D)
    # On utilise une différence de y talon entre événements (approx)
    stepG_norm = None
    stepD_norm = None

    if cG:
        i0, i1 = cG
        # “pas G” approximé : variation relative entre talon G (événement) et talon D (réf)
        stepG_norm = abs(hG[i1] - hD[i0])

    if cD:
        i0, i1 = cD
        stepD_norm = abs(hD[i1] - hG[i0])

    steps_norm = [v for v in [stepG_norm, stepD_norm] if v is not None]
    if not steps_norm:
        return None, None, None, None, None

    # Conversion en cm via facteur stature (approx)
    # 0.53 = ordre de grandeur talon->tête / stature (approx adulte)
    scale = taille_cm / 0.53

    stepG_cm = stepG_norm * scale if stepG_norm is not None else None
    stepD_cm = stepD_norm * scale if stepD_norm is not None else None

    step_mean_cm = float(np.mean([v for v in [stepG_cm, stepD_cm] if v is not None]))
    step_std_cm = float(np.std([v for v in [stepG_cm, stepD_cm] if v is not None]))

    step_asym = asym_percent(stepG_cm, stepD_cm)

    return step_mean_cm, step_std_cm, stepG_cm, stepD_cm, step_asym

# ==============================
# PDF EXPORT
# ==============================
def export_pdf(patient, keyframe_path, figures, table_data, annotated_images, step_info=None, asym_table=None):
    out_path = os.path.join(tempfile.gettempdir(), f"GaitScan_{patient['nom']}_{patient['prenom']}.pdf")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.7*cm, rightMargin=1.7*cm,
        topMargin=1.7*cm, bottomMargin=1.7*cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique</b>", styles["Title"]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Angle de film :</b> {patient.get('camera','N/A')}<br/>"
        f"<b>Affichage phases :</b> {patient.get('phase','N/A')}<br/>"
        f"<b>Taille :</b> {patient.get('taille_cm','N/A')} cm",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.35*cm))

    # --- Step length block
    if step_info is not None:
        story.append(Paragraph("<b>Paramètres spatio-temporels (estimation)</b>", styles["Heading2"]))
        story.append(Paragraph(
            f"<b>Longueur de pas moyenne :</b> {step_info['mean']:.1f} cm<br/>"
            f"<b>Variabilité :</b> ± {step_info['std']:.1f} cm<br/>"
            + (f"<b>Pas G :</b> {step_info['G']:.1f} cm &nbsp;&nbsp; <b>Pas D :</b> {step_info['D']:.1f} cm<br/>" if step_info.get("G") is not None and step_info.get("D") is not None else "")
            + (f"<b>Asymétrie pas (G/D) :</b> {step_info['asym']:.1f} %<br/>" if step_info.get("asym") is not None else "")
            + "<i>Mesure monocaméra 2D sans calibration métrique : valeurs estimées.</i>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.25*cm))

    # --- Asymmetry block (angles)
    if asym_table:
        story.append(Paragraph("<b>Asymétries droite/gauche (angles)</b>", styles["Heading2"]))
        t = Table([["Mesure", "Moy G", "Moy D", "Asym %"]] + asym_table,
                  colWidths=[6*cm, 3*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.7,colors.black),
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("ALIGN",(1,1),(-1,-1),"CENTER")
        ]))
        story.append(t)
        story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph("<b>Image clé</b>", styles["Heading2"]))
    story.append(PDFImage(keyframe_path, width=16*cm, height=8*cm))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("<b>Analyse articulaire</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))
    for joint, figpath in figures.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles["Heading3"]))
        story.append(PDFImage(figpath, width=16*cm, height=6*cm))
        story.append(Spacer(1, 0.3*cm))

    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("<b>Synthèse (°)</b>", styles["Heading2"]))

    table = Table([["Mesure", "Min", "Moyenne", "Max"]] + table_data,
                  colWidths=[7*cm, 3*cm, 3*cm, 3*cm])
    table.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.7,colors.black),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("ALIGN",(1,1),(-1,-1),"CENTER")
    ]))
    story.append(table)

    if annotated_images:
        story.append(PageBreak())
        story.append(Paragraph("<b>Images annotées (angles)</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))
        for img in annotated_images:
            story.append(PDFImage(img, width=16*cm, height=8*cm))
            story.append(Spacer(1, 0.25*cm))

    doc.build(story)
    return out_path

# ==============================
# PDF VIEW + PRINT (browser-side)
# ==============================
def pdf_viewer_with_print(pdf_bytes: bytes, height=800):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f"""
    <div style="display:flex; gap:12px; align-items:center; margin: 6px 0 10px 0;">
      <button onclick="printPdf()" style="padding:10px 14px; font-size:16px; cursor:pointer;">
        🖨️ Imprimer le rapport
      </button>
      <span style="opacity:0.7;">(ouvre la boîte d’impression du navigateur)</span>
    </div>
    <iframe id="pdfFrame" src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" style="border:1px solid #ddd; border-radius:8px;"></iframe>
    <script>
      function printPdf() {{
        const iframe = document.getElementById('pdfFrame');
        iframe.contentWindow.focus();
        iframe.contentWindow.print();
      }}
    </script>
    """
    components.html(html, height=height+80, scrolling=True)

# ==============================
# UI
# ==============================
with st.sidebar:
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")
    camera_pos = st.selectbox("Angle de film", ["Devant","Droite","Gauche"])
    phase_cote = st.selectbox("Phases", ["Aucune","Droite","Gauche","Les deux"])
    smooth = st.slider("Lissage", 0, 10, 3)
    conf = st.slider("Seuil confiance", 0.1, 0.9, 0.3, 0.05)

    # ✅ AJOUT : taille patient
    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)

video = st.file_uploader("Vidéo", ["mp4","avi","mov"])

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read()); tmp.close()

    data, heelG, heelD, frames = process_video(tmp.name, conf)
    os.unlink(tmp.name)

    # Phases (double)
    phases = []
    if phase_cote in ["Gauche","Les deux"]:
        c = detect_cycle(heelG)
        if c: phases.append((*c, "orange"))
    if phase_cote in ["Droite","Les deux"]:
        c = detect_cycle(heelD)
        if c: phases.append((*c, "blue"))

    # ==============================
    # LONGUEUR DE PAS (estimation)
    # ==============================
    step_mean, step_std, stepG_cm, stepD_cm, step_asym = compute_step_length_cm(heelG, heelD, float(taille_cm))

    st.subheader("📏 Paramètres spatio-temporels")
    if step_mean is not None:
        st.write(f"**Longueur de pas moyenne :** {step_mean:.1f} cm")
        st.write(f"**Variabilité (±1σ) :** {step_std:.1f} cm")
        if stepG_cm is not None and stepD_cm is not None:
            st.write(f"**Pas G :** {stepG_cm:.1f} cm — **Pas D :** {stepD_cm:.1f} cm")
        if step_asym is not None:
            st.write(f"**Asymétrie pas (G/D) :** {step_asym:.1f} %")
        st.caption("Estimation monocaméra 2D sans calibration métrique (échelle basée sur la taille).")
    else:
        st.warning("Longueur de pas non calculable (talons insuffisamment détectés).")

    # keyframe (milieu)
    keyframe_path = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(keyframe_path, frames[len(frames)//2])

    # Graphs + save figures for PDF
    figures = {}
    table_data = []
    asym_rows = []  # pour PDF

    for joint in ["Hanche","Genou","Cheville"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios":[2,1]})

        g_raw = np.array(data[f"{joint} G"], dtype=float)
        d_raw = np.array(data[f"{joint} D"], dtype=float)

        # filtrage (NaN->0 pour filtrer, mais stats sur frames valides)
        g = bandpass(np.nan_to_num(g_raw, nan=0.0), smooth)
        d = bandpass(np.nan_to_num(d_raw, nan=0.0), smooth)

        ax1.plot(g, label="Gauche", color="red")
        ax1.plot(d, label="Droite", color="blue")
        for c0, c1, col in phases:
            ax1.axvspan(c0, c1, color=col, alpha=0.3)
        ax1.set_title(f"{joint} – Analyse")
        ax1.legend()

        ax2.plot(norm_curve(joint, len(g)), color="green")
        ax2.set_title("Norme")

        st.pyplot(fig)

        fig_path = os.path.join(tempfile.gettempdir(), f"{joint}_plot.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        figures[joint] = fig_path

        # stats (filtré, mais seulement sur indices non NaN du raw)
        def stats(arr_filtered, arr_raw):
            mask = ~np.isnan(arr_raw)
            if mask.sum() == 0:
                return np.nan, np.nan, np.nan, None
            vals = arr_filtered[mask]
            return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), float(np.mean(vals))

        gmin, gmean, gmax, gmean_only = stats(g, g_raw)
        dmin, dmean, dmax, dmean_only = stats(d, d_raw)

        table_data.append([f"{joint} Gauche", f"{gmin:.1f}", f"{gmean:.1f}", f"{gmax:.1f}"])
        table_data.append([f"{joint} Droite", f"{dmin:.1f}", f"{dmean:.1f}", f"{dmax:.1f}"])

        # asymétrie (%) sur moyennes
        a = asym_percent(gmean_only, dmean_only)
        if a is None:
            asym_rows.append([joint, f"{gmean_only:.1f}" if gmean_only is not None else "NA",
                              f"{dmean_only:.1f}" if dmean_only is not None else "NA",
                              "NA"])
        else:
            asym_rows.append([joint, f"{gmean_only:.1f}", f"{dmean_only:.1f}", f"{a:.1f}"])

    # Affichage asymétries dans l'app
    st.subheader("↔️ Asymétries droite/gauche (angles)")
    for row in asym_rows:
        st.write(f"**{row[0]}** — Moy G: {row[1]}° | Moy D: {row[2]}° | Asym: {row[3]}%")

    # ==============================
    # CAPTURES ANNOTÉES
    # ==============================
    st.subheader("📸 Captures annotées (angles)")
    num_photos = st.slider("Nombre d'images extraites", 1, 10, 3)
    total_frames = len(frames)
    idxs = np.linspace(0, total_frames-1, num_photos, dtype=int)

    annotated_images = []
    for i, idx in enumerate(idxs):
        frame = frames[idx]
        kp = detect_pose(frame)  # recalcul uniquement pour ces frames
        ann = annotate_frame(frame, kp, conf=conf)

        out_img = os.path.join(tempfile.gettempdir(), f"annotated_{i}.png")
        cv2.imwrite(out_img, ann)
        annotated_images.append(out_img)

        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"Image annotée {i+1} (frame {idx})")

    # ==============================
    # PDF
    # ==============================
    step_info = None
    if step_mean is not None:
        step_info = {
            "mean": step_mean,
            "std": step_std,
            "G": stepG_cm,
            "D": stepD_cm,
            "asym": step_asym
        }

    pdf_path = export_pdf(
        patient={"nom": nom, "prenom": prenom, "camera": camera_pos, "phase": phase_cote, "taille_cm": int(taille_cm)},
        keyframe_path=keyframe_path,
        figures=figures,
        table_data=table_data,
        annotated_images=annotated_images,
        step_info=step_info,
        asym_table=asym_rows
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.success("✅ Rapport généré")
    st.download_button(
        "📄 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=f"GaitScan_{nom}_{prenom}.pdf",
        mime="application/pdf"
    )


