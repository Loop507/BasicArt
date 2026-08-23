"""
app_BASICART.py
BasicArt // Loop507
------------------------------------------------------------
Genera un video generativo animato a partire da un brano audio,
usando un motore grafico ispirato al BASIC primitivo anni '80
(FOR/NEXT, PLOT, funzioni trigonometriche) pilotato da analisi
DSP pura del segnale (librosa). Nessun modello AI/neurale.

Due motori grafici disponibili (stessa famiglia di equazioni,
proiezioni diverse):
- "Ellisse"  : x=f(t2), y=f(t1)                  (coordinate cartesiane)
- "Loto"     : r=f(t2), a=f(t1), x=r*cos(a) ...  (coordinate polari)

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
MAX_DURATION_S = 240             # cap di sicurezza per il rendering (4 minuti)

RISOLUZIONI = {
    "16:9  (1280x720)": (1280, 720),
    "9:16  (720x1280)": (720, 1280),
}

FORME = ["Ellisse (cartesiana)", "Loto (polare)"]

st.set_page_config(page_title="BasicArt // Loop507", layout="centered")


# ----------------------------------------------------------------------
# DSP: helper generici
# ----------------------------------------------------------------------
def _adatta(arr, n):
    """Ricampiona un array 1D a n punti (nearest-index)."""
    if len(arr) == 0:
        return np.zeros(n)
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx]


def _norm(arr):
    """Normalizza in [0,1]."""
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 1e-9 else np.zeros_like(arr)


def _smussa(arr, alpha=0.12):
    """Media mobile esponenziale: evita salti bruschi frame-per-frame
    che rendono il disegno caotico invece che fluido."""
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _bande_spettrali(y, sr, hop_length, n_frames):
    """Energia in 3 bande di frequenza (bassi/medi/alti) via STFT.
    Analisi DSP pura, nessun modello AI."""
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)

    m_bassi = freqs < 250
    m_medi = (freqs >= 250) & (freqs < 2000)
    m_alti = freqs >= 2000

    def energia(mask):
        return S[mask].mean(axis=0) if mask.any() else np.zeros(S.shape[1])

    bassi = _adatta(energia(m_bassi), n_frames)
    medi = _adatta(energia(m_medi), n_frames)
    alti = _adatta(energia(m_alti), n_frames)
    return _norm(bassi), _norm(medi), _norm(alti)


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

    rms = _norm(_adatta(rms, n_frames))
    centroid = _norm(_adatta(centroid, n_frames))
    onset_env = _norm(_adatta(onset_env, n_frames))

    bassi, medi, alti = _bande_spettrali(y, sr, hop_length, n_frames)

    return {
        "rms": _smussa(rms),
        "centroid": _smussa(centroid),
        "onset": _smussa(onset_env, alpha=0.20),   # onset resta un po' piu' reattivo
        "bassi": _smussa(bassi),
        "medi": _smussa(medi),
        "alti": _smussa(alti),
        "durata": durata,
        "n_frames": n_frames,
        "sr": sr,
    }


# ----------------------------------------------------------------------
# MOTORE GRAFICO "BASIC PRIMITIVO"
# ------------------------------------------------------------
# Pseudo-programma di riferimento (stile BASIC anni '80):
#
#   ELLISSE (cartesiana):
#     10 FOR T1 = 0 TO 75 STEP DT
#     20   F = AMP * (SIN(PI*T2) + SIN(K*T2))
#     30   PLOT CX+F(T2), CY+F(T1)
#     40   T2 = T2 + DT
#     50 NEXT T1
#
#   LOTO (polare):
#     10 FOR T1 = 0 TO 300 STEP DT
#     20   R = F(T2) : A = F(T1)
#     30   PLOT CX+AMP*R*COS(A), CY+AMP*R*SIN(A)
#     40   T2 = T2 + DT
#     50 NEXT T1
#
# Ogni parametro (AMP, K, DT, OFFSET) e' pilotato da una feature
# audio diversa: energia globale + bassi -> ampiezza, medi -> una
# frequenza secondaria, alti/centroide -> l'altra, onset -> ritmo
# di campionamento e deriva dell'offset.
# ----------------------------------------------------------------------
def _parametri_da_audio(feat, i, raggio):
    """Calcola i parametri del motore combinando piu' feature audio,
    cosi' brani diversi (per energia, timbro, bande di frequenza)
    producono disegni chiaramente distinti."""
    rms = feat["rms"][i]
    centroid = feat["centroid"][i]
    onset = feat["onset"][i]
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]

    energia = 0.6 * rms + 0.4 * bassi
    amp = raggio * (0.55 + 0.45 * energia)

    k1 = np.clip(2.0 + 3.0 * centroid + 2.0 * alti, 2.0, 7.0)   # timbro/alti
    k2 = np.clip(2.0 + 3.0 * medi, 2.0, 7.0)                     # medi

    # per il pattern Loto (polare) le due componenti r e a devono condividere
    # la STESSA frequenza secondaria (come nel riferimento BASIC originale,
    # k=3.5 fisso per entrambe): usare due k diversi rompe la correlazione
    # armonica e trasforma i petali puliti in un groviglio denso stile Lissajous
    k_loto = np.clip(2.6 + 1.2 * centroid + 0.8 * alti + 0.6 * medi, 2.6, 4.4)

    intensita = 0.65 + 0.35 * onset
    return amp, k1, k2, k_loto, onset, intensita


def disegna_ellisse(canvas, t_frame, feat, i, cx, cy, raggio, colore_fg, t1_arr):
    """Pattern cartesiano: x=f(t2), y=f(t1)."""
    amp, k1, _k2, _k_loto, onset, intensita = _parametri_da_audio(feat, i, raggio)

    dt = 0.02 + 0.008 * onset
    offset = t_frame * 0.004 + onset * 0.3

    t1_vals = t1_arr * dt
    t2_vals = offset + t1_arr * dt

    def f(x):
        return amp * (np.sin(np.pi * x) + np.sin(k1 * x))

    xs = (cx + f(t2_vals)).astype(np.int32)
    ys = (cy + f(t1_vals)).astype(np.int32)

    colore = tuple(min(int(c * intensita), 255) for c in colore_fg)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=colore, thickness=2, lineType=cv2.LINE_AA)
    return canvas


def disegna_loto(canvas, t_frame, feat, i, cx, cy, raggio, colore_fg, t1_arr):
    """Pattern polare: r=f(t2), a=f(t1), x=r*cos(a), y=r*sin(a).
    r e a condividono la stessa frequenza secondaria k (come nel BASIC
    originale) per mantenere la simmetria a petali. A differenza
    dell'Ellisse, qui i punti vengono tracciati SPARSI (stile PLOT del
    BASIC originale) e non collegati con linee: la struttura a petali
    del "fiore di loto" emerge dall'accumulo di punti isolati nel tempo,
    non da una curva continua."""
    amp, _k1, _k2, k, onset, intensita = _parametri_da_audio(feat, i, raggio)

    dt = 0.10 + 0.05 * onset
    offset = t_frame * 0.002 + onset * 0.10

    t1_vals = t1_arr * dt
    t2_vals = offset + t1_arr * dt

    def f(x):
        return np.sin(np.pi * x) + np.sin(k * x)

    r = f(t2_vals)          # bassi/rms -> "respiro" radiale
    a = f(t1_vals)          # stessa k -> struttura a petali coerente

    xs = (cx + amp * r * np.cos(a)).astype(np.int32)
    ys = (cy + amp * r * np.sin(a)).astype(np.int32)

    h, w = canvas.shape[:2]
    dentro = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[dentro], ys[dentro]

    colore = np.array([min(int(c * intensita), 255) for c in colore_fg], dtype=np.uint8)
    canvas[ys, xs] = colore   # PLOT: punti isolati, non connessi

    return canvas


MOTORI = {
    "Ellisse (cartesiana)": {"funzione": disegna_ellisse, "n_step": 900, "fade": 0.90},
    "Loto (polare)": {"funzione": disegna_loto, "n_step": 3300, "fade": 0.80},
}


def genera_video(feat, path_out, width, height, colore_bg, colore_fg, forma, fps=FPS, seed=507):
    np.random.seed(seed)
    cx, cy = width // 2, height // 2
    raggio = min(width, height) * 0.22   # proporzione calibrata sul riferimento BASIC originale

    motore = MOTORI[forma]
    disegna = motore["funzione"]
    n_step = motore["n_step"]
    fade_alpha = motore["fade"]
    t1_arr = np.arange(n_step, dtype=np.float64)   # indice pre-calcolato una sola volta

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    bg = np.array(colore_bg, dtype=np.float32)

    n_frames = feat["n_frames"]
    progress = st.progress(0, text="RENDER :: generazione frame in corso...")

    for i in range(n_frames):
        # fade verso il nero (scia stile fosfori)
        canvas = canvas * fade_alpha + bg * (1 - fade_alpha)

        canvas_u8 = canvas.astype(np.uint8)
        canvas_u8 = disegna(
            canvas_u8, t_frame=i, feat=feat, i=i,
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

    forma = st.selectbox("Forma :: Shape", FORME)

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
        n_step_stimato = MOTORI[forma]["n_step"]
        fattore_forma = n_step_stimato / 900
        st.caption(
            "Tempo di render stimato :: Estimated render time ~ "
            f"{(width * height) / (1280 * 720) * 0.55 * fattore_forma:.0f}s ogni minuto di brano "
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
                    f"FORMATO :: {width}x{height}  |  "
                    f"FORMA :: {forma}"
                )

                path_video_muto = os.path.join(tmp, "video_muto.mp4")
                genera_video(feat, path_video_muto, width, height, colore_bg, colore_fg, forma)

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
