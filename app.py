"""
app_BASICART.py
BasicArt // Loop507
------------------------------------------------------------
Genera un video generativo animato a partire da un brano audio,
usando un motore grafico ispirato al BASIC primitivo anni '80
(FOR/NEXT, PLOT, funzioni trigonometriche) pilotato da analisi
DSP pura del segnale (librosa). Nessun modello AI/neurale.

Loop507 protocol:
- py_compile / pyflakes zero warnings
- report bilingue IT/EN con formattazione "::"
- seed per riproducibilita'
- session_state per persistenza download
------------------------------------------------------------
"""

import os
import tempfile
import numpy as np
import cv2
import streamlit as st
import librosa
from moviepy import VideoFileClip, AudioFileClip

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FPS = 30
FADE_ALPHA = 0.90                # scia: quanto rimane del frame precedente
MAX_DURATION_S = 240             # cap di sicurezza per il rendering (4 minuti)

RISOLUZIONI = {
    "16:9  (1280x720)": (1280, 720),
    "9:16  (720x1280)": (720, 1280),
}

st.set_page_config(page_title="BasicArt // Loop507", layout="centered")


# ----------------------------------------------------------------------
# DSP: estrazione feature audio (pura, no AI)
# ----------------------------------------------------------------------
def analizza_audio(path_audio, fps, durata_max=MAX_DURATION_S):
    """Estrae feature frame-per-frame sincronizzate al framerate video."""
    y, sr = librosa.load(path_audio, sr=None, mono=True, duration=durata_max)
    durata = len(y) / sr
    n_frames = max(1, int(durata * fps))

    hop_length = max(1, int(sr / fps))

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    def adatta(arr, n):
        if len(arr) == 0:
            return np.zeros(n)
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return arr[idx]

    rms = adatta(rms, n_frames)
    centroid = adatta(centroid, n_frames)
    onset_env = adatta(onset_env, n_frames)

    def norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 1e-9 else np.zeros_like(arr)

    return {
        "rms": norm(rms),
        "centroid": norm(centroid),
        "onset": norm(onset_env),
        "durata": durata,
        "n_frames": n_frames,
        "sr": sr,
    }


# ----------------------------------------------------------------------
# MOTORE GRAFICO "BASIC PRIMITIVO"
# ------------------------------------------------------------
# Pseudo-programma di riferimento (stile BASIC anni '80):
#
#   10 FOR T1 = 0 TO 75 STEP DT
#   20   F = 250 + AMP * (SIN(PI*T2) + SIN(K*T2))
#   30   PLOT F(T2), F(T1)
#   40   T2 = T2 + DT
#   50 NEXT T1
#
# Qui AMP, K, DT e OFFSET sono modulati dalle feature audio
# invece di crescere in modo fisso ad ogni frame.
# ----------------------------------------------------------------------
def disegna_frame(canvas, t_frame, feat_rms, feat_centroid, feat_onset, cx, cy, raggio, colore_fg, t1_arr):
    """Disegna un frame del pattern non-periodico pilotato dall'audio (vettorizzato NumPy)."""
    # canvas gia' sfumato (fade) prima della chiamata

    amp = raggio * (0.35 + 0.65 * feat_rms)            # energia -> ampiezza
    k = 2.0 + 8.0 * feat_centroid                       # brillantezza -> frequenza secondaria
    dt = 0.02 + 0.015 * feat_onset                      # onset/beat -> passo di campionamento
    offset = t_frame * 0.004 + feat_onset * 0.6         # deriva lenta + scatto sui beat

    t1_vals = t1_arr * dt                               # t1 = i * dt, vettorizzato
    t2_vals = offset + t1_arr * dt                      # t2 = offset + i * dt, vettorizzato

    def f(x):
        return amp * (np.sin(np.pi * x) + np.sin(k * x))

    xs = (cx + f(t2_vals)).astype(np.int32)
    ys = (cy + f(t1_vals)).astype(np.int32)

    h, w = canvas.shape[:2]
    dentro = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[dentro], ys[dentro]

    intensita = 0.70 + 0.30 * feat_onset
    colore = np.array([min(int(c * intensita), 255) for c in colore_fg], dtype=np.uint8)

    canvas[ys, xs] = colore   # scrittura vettorizzata dei punti (no loop Python)

    return canvas


def genera_video(feat, path_out, width, height, colore_bg, colore_fg, fps=FPS, seed=507):
    np.random.seed(seed)
    cx, cy = width // 2, height // 2
    raggio = min(width, height) * 0.30
    n_step = 900
    t1_arr = np.arange(n_step, dtype=np.float64)   # indice pre-calcolato una sola volta

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    bg = np.array(colore_bg, dtype=np.float32)

    n_frames = feat["n_frames"]
    progress = st.progress(0, text="RENDER :: generazione frame in corso...")

    for i in range(n_frames):
        # fade verso il nero (scia stile fosfori)
        canvas = canvas * FADE_ALPHA + bg * (1 - FADE_ALPHA)

        canvas_u8 = canvas.astype(np.uint8)
        canvas_u8 = disegna_frame(
            canvas_u8,
            t_frame=i,
            feat_rms=feat["rms"][i],
            feat_centroid=feat["centroid"][i],
            feat_onset=feat["onset"][i],
            cx=cx, cy=cy, raggio=raggio,
            colore_fg=colore_fg, t1_arr=t1_arr,
        )
        canvas = canvas_u8.astype(np.float32)

        writer.write(canvas_u8)

        if i % 5 == 0 or i == n_frames - 1:
            progress.progress((i + 1) / n_frames, text=f"RENDER :: frame {i+1}/{n_frames}")

    writer.release()
    progress.empty()


def mux_audio(path_video_muto, path_audio, path_out, durata):
    video_clip = VideoFileClip(path_video_muto)
    audio_clip = AudioFileClip(path_audio).subclipped(0, durata)
    final = video_clip.with_audio(audio_clip)
    final.write_videofile(
        path_out, codec="libx264", audio_codec="aac",
        fps=FPS, logger=None
    )
    video_clip.close()
    audio_clip.close()
    final.close()


# ----------------------------------------------------------------------
# UI STREAMLIT
# ----------------------------------------------------------------------
def main():
    st.title("BasicArt // Loop507")
    st.caption(
        "Disegni generativi animati da un brano audio, motore ispirato al BASIC "
        "primitivo anni '80 :: DSP puro, nessuna AI\n\n"
        "Generative drawings animated from an audio track, engine inspired by "
        "primitive 1980s BASIC :: pure DSP, no AI"
    )

    file_audio = st.file_uploader(
        "Carica un brano (WAV/MP3) :: Upload a track (WAV/MP3)",
        type=["wav", "mp3", "ogg", "flac"],
    )

    col1, col2 = st.columns(2)
    with col1:
        risoluzione_label = st.selectbox(
            "Formato :: Aspect ratio", list(RISOLUZIONI.keys())
        )
    with col2:
        st.caption(f"Durata max :: {MAX_DURATION_S // 60} minuti")

    col3, col4 = st.columns(2)
    with col3:
        hex_bg = st.color_picker("Colore sfondo :: Background color", "#0a0a0a")
    with col4:
        hex_fg = st.color_picker("Colore animazione :: Animation color", "#ebebeb")

    def hex_a_bgr(hex_str):
        hex_str = hex_str.lstrip("#")
        r, g, b = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    colore_bg = hex_a_bgr(hex_bg)
    colore_fg = hex_a_bgr(hex_fg)
    width, height = RISOLUZIONI[risoluzione_label]

    if file_audio is not None:
        st.caption(
            "Tempo di render stimato :: Estimated render time ~ "
            f"{(width * height) / (1280 * 720) * 0.55:.0f}s ogni minuto di brano "
            "(indicativo, dipende dal server)"
        )
        if st.button("Genera video :: Generate video", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                path_in = os.path.join(tmp, file_audio.name)
                with open(path_in, "wb") as f:
                    f.write(file_audio.getbuffer())

                with st.spinner("Analisi DSP del brano :: DSP analysis..."):
                    feat = analizza_audio(path_in, FPS)

                st.info(
                    f"DURATA :: {feat['durata']:.1f}s  |  "
                    f"FRAME :: {feat['n_frames']}  |  "
                    f"SR :: {feat['sr']} Hz  |  "
                    f"FORMATO :: {width}x{height}"
                )

                path_video_muto = os.path.join(tmp, "video_muto.mp4")
                genera_video(feat, path_video_muto, width, height, colore_bg, colore_fg)

                path_finale = os.path.join(tmp, "basicart_output.mp4")
                with st.spinner("Muxaggio audio/video :: Audio/video muxing..."):
                    mux_audio(path_video_muto, path_in, path_finale, feat["durata"])

                with open(path_finale, "rb") as f:
                    video_bytes = f.read()

                st.session_state["basicart_video"] = video_bytes
                st.session_state["basicart_filename"] = "basicart_" + os.path.splitext(file_audio.name)[0] + ".mp4"

    if "basicart_video" in st.session_state:
        st.success("RENDER COMPLETATO :: Render complete")
        st.video(st.session_state["basicart_video"])
        st.download_button(
            "Scarica video :: Download video",
            data=st.session_state["basicart_video"],
            file_name=st.session_state["basicart_filename"],
            mime="video/mp4",
        )


if __name__ == "__main__":
    main()
