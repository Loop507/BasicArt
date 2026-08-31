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
    "1:1   (720x720)": (720, 720),
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


def _stima_bpm(y, sr):
    """Stima il tempo (BPM) del brano via beat tracking DSP (no AI).
    Il valore pilota la velocita' di rotazione/pulsazione del pattern:
    reinterpretazione originale, non presente negli esempi BASIC di
    riferimento — collega il "senso del tempo" del brano al motore."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception:
        bpm = 120.0
    if bpm <= 0:
        bpm = 120.0
    return bpm


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
    bpm = _stima_bpm(y, sr)

    return {
        "rms": _smussa(rms),
        "centroid": _smussa(centroid),
        "onset": _smussa(onset_env, alpha=0.20),   # onset resta un po' piu' reattivo
        "bassi": _smussa(bassi),
        "medi": _smussa(medi),
        "alti": _smussa(alti),
        "bpm": bpm,
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
def _parametri_da_audio(feat, i, t_frame, fps, reattivita=1.0):
    """Calcola i parametri del motore combinando piu' feature audio,
    cosi' brani diversi (per energia, timbro, bande di frequenza, tempo)
    producono disegni chiaramente distinti. 'reattivita' e' un guadagno
    globale (controllabile da UI) su quanto l'audio muove i parametri."""
    rms = feat["rms"][i]
    centroid = feat["centroid"][i]
    onset = feat["onset"][i] * reattivita
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]
    bpm = feat["bpm"]

    energia = (0.6 * rms + 0.4 * bassi) * reattivita

    # velocita' di rotazione/pulsazione ancorata al BPM del brano (reinterpretazione
    # originale: un brano a 160 BPM "gira" piu' veloce di una ballata a 70 BPM)
    velocita = np.clip(bpm / 120.0, 0.55, 1.9)

    # pulsazione "a respiro" sincronizzata al beat: una lieve oscillazione
    # dell'ampiezza al ritmo esatto del brano, indipendente dalla reattivita'
    # istantanea dell'onset — tocco creativo, non presente nei riferimenti BASIC
    fase_beat = 2 * np.pi * (bpm / 60.0) * (t_frame / fps)
    respiro = 1.0 + 0.06 * np.sin(fase_beat)

    fattore_ampiezza = np.clip((0.55 + 0.45 * energia) * respiro, 0.25, 2.0)

    k1 = np.clip(2.0 + (3.0 * centroid + 2.0 * alti) * reattivita, 2.0, 9.0)   # timbro/alti
    k2 = np.clip(2.0 + 3.0 * medi * reattivita, 2.0, 9.0)                       # medi

    # per il pattern Loto (polare) le due componenti r e a devono condividere
    # la STESSA frequenza secondaria (come nel riferimento BASIC originale,
    # k=3.5 fisso per entrambe): usare due k diversi rompe la correlazione
    # armonica e trasforma i petali puliti in un groviglio denso stile Lissajous
    k_loto = np.clip(2.6 + (1.2 * centroid + 0.8 * alti + 0.6 * medi) * reattivita, 2.2, 5.2)

    intensita = np.clip(0.88 + 0.12 * onset, 0.5, 1.0)   # baseline alta: il colore scelto
                                                            # resta vivo, l'onset aggiunge
                                                            # solo un lieve guizzo extra
    return fattore_ampiezza, k1, k2, k_loto, onset, intensita, velocita


def disegna_ellisse(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_fg, t1_arr, fps,
                     reattivita=1.0, spessore=2):
    """Pattern cartesiano: x=f(t2), y=f(t1). Scala anisotropica (raggio_x/
    raggio_y separati) per riempire il fotogramma invece di restare confinato
    al centro — reinterpretazione mia rispetto al riferimento (che usava un
    unico raggio isotropo su un canvas quadrato)."""
    fattore, k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )

    dt = (0.02 + 0.008 * onset) * velocita
    offset = t_frame * 0.004 * velocita + onset * 0.3

    t1_vals = t1_arr * dt
    t2_vals = offset + t1_arr * dt

    def f(x):
        return np.sin(np.pi * x) + np.sin(k1 * x)

    fx = f(t2_vals) * fattore
    fy = f(t1_vals) * fattore

    xs = (cx + raggio_x * fx).astype(np.int32)
    ys = (cy + raggio_y * fy).astype(np.int32)

    colore = tuple(min(int(c * intensita), 255) for c in colore_fg)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=colore, thickness=spessore, lineType=cv2.LINE_AA)
    return canvas


def disegna_loto(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_fg, t1_arr, fps,
                  reattivita=1.0, spessore=2):
    """Pattern polare: r=f(t2), a=f(t1), x=r*cos(a), y=r*sin(a).
    r e a condividono la stessa frequenza secondaria k (come nel BASIC
    originale) per mantenere la simmetria a petali. A differenza
    dell'Ellisse, qui i punti vengono tracciati SPARSI (stile PLOT del
    BASIC originale) e non collegati con linee: la struttura a petali
    del "fiore di loto" emerge dall'accumulo di punti isolati nel tempo,
    non da una curva continua. Scala anisotropica per riempire meglio
    formati rettangolari (16:9, 9:16) invece di restare un piccolo
    medaglione centrale. 'spessore' dilata leggermente i punti (via
    maschera + dilate) per renderli visibili anche a bassa densita'."""
    fattore, _k1, _k2, k, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )

    dt = (0.10 + 0.05 * onset) * velocita
    offset = t_frame * 0.002 * velocita + onset * 0.10

    t1_vals = t1_arr * dt
    t2_vals = offset + t1_arr * dt

    def f(x):
        return np.sin(np.pi * x) + np.sin(k * x)

    r = f(t2_vals) * fattore
    a = f(t1_vals)

    xs = (cx + raggio_x * r * np.cos(a)).astype(np.int32)
    ys = (cy + raggio_y * r * np.sin(a)).astype(np.int32)

    h, w = canvas.shape[:2]
    dentro = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[dentro], ys[dentro]

    colore = np.array([min(int(c * intensita), 255) for c in colore_fg], dtype=np.uint8)

    if spessore <= 1:
        canvas[ys, xs] = colore   # PLOT: punti isolati, non connessi
    else:
        # dilatazione vettorizzata: punti piu' spessi e visibili senza
        # disegnare migliaia di cerchi singolarmente (resta veloce)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ys, xs] = 255
        kernel = np.ones((spessore, spessore), np.uint8)
        mask = cv2.dilate(mask, kernel)
        canvas[mask > 0] = colore

    return canvas


MOTORI = {
    "Ellisse (cartesiana)": {"funzione": disegna_ellisse, "n_step": 900, "fade": 0.90},
    "Loto (polare)": {"funzione": disegna_loto, "n_step": 3300, "fade": 0.80},
}


def genera_video(feat, path_out, width, height, colore_bg, colore_fg, forma, fps=FPS, seed=507,
                  densita=1.0, spessore=2, reattivita=1.0):
    np.random.seed(seed)
    cx, cy = width // 2, height // 2
    # scala anisotropica sui due assi (invece di un unico raggio isotropo):
    # il pattern si estende a riempire il fotogramma, non solo il centro
    raggio_x = width * 0.34
    raggio_y = height * 0.34

    motore = MOTORI[forma]
    disegna = motore["funzione"]
    n_step = max(100, int(motore["n_step"] * densita))
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
            cx=cx, cy=cy, raggio_x=raggio_x, raggio_y=raggio_y,
            colore_fg=colore_fg, t1_arr=t1_arr, fps=fps,
            reattivita=reattivita, spessore=spessore,
        )
        canvas = canvas_u8.astype(np.float32)

        writer.write(canvas_u8)

        if i % 5 == 0 or i == n_frames - 1:
            progress.progress((i + 1) / n_frames, text=f"RENDER :: frame {i+1}/{n_frames}")

    writer.release()
    progress.empty()


def genera_anteprima(feat, width, height, colore_bg, colore_fg, forma, densita, spessore,
                      reattivita, seed=507, finestra_s=4.0, fps=FPS):
    """Genera un'immagine statica che mostra come apparirebbe il pattern al
    picco energetico del brano (RMS+bassi massimi). Simula solo la finestra
    di pochi secondi che precede il picco (la scia decade rapidamente, quindi
    il resto del brano non influisce sul risultato) — abbastanza veloce da
    rigenerare ad ogni modifica degli slider, senza incidere sulla DSP gia'
    calcolata ne' scrivere un video."""
    np.random.seed(seed)
    cx, cy = width // 2, height // 2
    raggio_x = width * 0.34
    raggio_y = height * 0.34

    motore = MOTORI[forma]
    disegna = motore["funzione"]
    n_step = max(100, int(motore["n_step"] * densita))
    fade_alpha = motore["fade"]
    t1_arr = np.arange(n_step, dtype=np.float64)

    energia = 0.6 * feat["rms"] + 0.4 * feat["bassi"]
    # esclude un margine ai bordi: l'attacco/coda del brano puo' generare un
    # picco artificiale nella STFT (transiente di inizio/fine file) che non
    # rappresenta il vero apice energetico del brano
    margine = min(int(0.5 * fps), max(0, len(energia) // 2 - 1))
    if len(energia) > 2 * margine:
        i_picco = margine + int(np.argmax(energia[margine:len(energia) - margine]))
    else:
        i_picco = int(np.argmax(energia))
    i_inizio = max(0, i_picco - int(finestra_s * fps))

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    bg = np.array(colore_bg, dtype=np.float32)

    for i in range(i_inizio, i_picco + 1):
        canvas = canvas * fade_alpha + bg * (1 - fade_alpha)
        canvas_u8 = canvas.astype(np.uint8)
        canvas_u8 = disegna(
            canvas_u8, t_frame=i, feat=feat, i=i,
            cx=cx, cy=cy, raggio_x=raggio_x, raggio_y=raggio_y,
            colore_fg=colore_fg, t1_arr=t1_arr, fps=fps,
            reattivita=reattivita, spessore=spessore,
        )
        canvas = canvas_u8.astype(np.float32)

    return canvas.astype(np.uint8), i_picco


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


def genera_report(nome_file, forma, width, height, risoluzione_label, feat,
                   hex_bg, hex_fg, densita, spessore, reattivita, seed):
    """Report bilingue IT/EN stile Loop507 (sezioni complete separate,
    formattazione '::', firma e hashtag finali). Restituisce sia la
    versione da mostrare in chat (con code-fence) sia il testo grezzo
    per il download."""
    separatore = "------------------------------------------------------------"

    it = (
        ":: BASICART // LOOP507 — REPORT ::\n"
        f"{separatore}\n"
        f"FILE            :: {nome_file}\n"
        f"FORMA           :: {forma}\n"
        f"FORMATO         :: {risoluzione_label}\n"
        f"DURATA          :: {feat['durata']:.1f}s\n"
        f"FRAME           :: {feat['n_frames']} @ {FPS}fps\n"
        f"SAMPLE RATE     :: {feat['sr']} Hz\n"
        f"BPM STIMATO     :: {feat['bpm']:.0f}\n"
        f"COLORE SFONDO   :: {hex_bg}\n"
        f"COLORE ANIM.    :: {hex_fg}\n"
        f"DENSITA'        :: {densita:.2f}x\n"
        f"SPESSORE        :: {spessore}\n"
        f"REATTIVITA'     :: {reattivita:.2f}x\n"
        f"SEED            :: {seed}\n"
        f"{separatore}\n"
        "DSP :: RMS, spectral centroid, onset strength, bande bassi/medi/\n"
        "       alti (STFT), beat tracking — nessun modello AI/neurale.\n"
        f"{separatore}\n"
        "Regia e Algoritmo :: Loop507\n"
        "L'audio e' stato scomposto in frequenze. Il codice ne ha ridisegnato la forma."
    )

    en = (
        ":: BASICART // LOOP507 — REPORT ::\n"
        f"{separatore}\n"
        f"FILE            :: {nome_file}\n"
        f"SHAPE           :: {forma}\n"
        f"RESOLUTION      :: {risoluzione_label}\n"
        f"DURATION        :: {feat['durata']:.1f}s\n"
        f"FRAMES          :: {feat['n_frames']} @ {FPS}fps\n"
        f"SAMPLE RATE     :: {feat['sr']} Hz\n"
        f"ESTIMATED BPM   :: {feat['bpm']:.0f}\n"
        f"BACKGROUND COLOR:: {hex_bg}\n"
        f"ANIMATION COLOR :: {hex_fg}\n"
        f"DENSITY         :: {densita:.2f}x\n"
        f"THICKNESS       :: {spessore}\n"
        f"REACTIVITY      :: {reattivita:.2f}x\n"
        f"SEED            :: {seed}\n"
        f"{separatore}\n"
        "DSP :: RMS, spectral centroid, onset strength, low/mid/high\n"
        "       bands (STFT), beat tracking — no AI/neural model.\n"
        f"{separatore}\n"
        "Direction & Algorithm :: Loop507\n"
        "The audio was broken down into frequencies. The code redrew its shape."
    )

    tag_forma = forma.split(" ")[0]   # "Ellisse" o "Loto"
    hashtags = (
        f"#Loop507 #BasicArt #{tag_forma} #GenerativeArt #AudioReactive "
        f"#DSP #PureDSP #BPM{feat['bpm']:.0f}"
    )

    testo_grezzo = f"{it}\n\n{en}\n\n{hashtags}"
    testo_markdown = "```\n" + testo_grezzo + "\n```"
    return testo_markdown, testo_grezzo


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
        hex_fg = st.color_picker("Colore animazione :: Animation color", "#ffffff")

    with st.expander("Controlli avanzati :: Advanced controls"):
        densita = st.slider(
            "Densita' punti :: Point density", 0.5, 2.0, 1.0, 0.1,
            help="Quanti punti per frame :: piu' alto = disegno piu' ricco/pieno"
        )
        spessore = st.slider(
            "Spessore :: Thickness", 1, 5, 2, 1,
            help="Spessore linee (Ellisse) o dimensione punti (Loto)"
        )
        reattivita = st.slider(
            "Reattivita' audio :: Audio reactivity", 0.5, 2.0, 1.2, 0.1,
            help="Quanto l'audio influenza ampiezza/frequenze del pattern"
        )

    def hex_a_bgr(hex_str):
        hex_str = hex_str.lstrip("#")
        r, g, b = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    colore_bg = hex_a_bgr(hex_bg)
    colore_fg = hex_a_bgr(hex_fg)
    width, height = RISOLUZIONI[risoluzione_label]

    if file_audio is not None:
        # L'audio caricato viene salvato una sola volta in un file persistente
        # (non nella TemporaryDirectory del bottone, che si cancella subito)
        # cosi' l'anteprima puo' rigenerarsi ad ogni slider senza rifare
        # l'analisi DSP o il salvataggio del file ogni volta.
        chiave_file = f"{file_audio.name}:{file_audio.size}"
        if st.session_state.get("basicart_feat_key") != chiave_file:
            path_persistente = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file_audio.name)[1]
            ).name
            with open(path_persistente, "wb") as f:
                f.write(file_audio.getbuffer())
            with st.spinner("Analisi DSP del brano :: DSP analysis..."):
                feat = analizza_audio(path_persistente, FPS)
            st.session_state["basicart_feat_key"] = chiave_file
            st.session_state["basicart_feat_value"] = feat
            st.session_state["basicart_audio_path"] = path_persistente
        else:
            feat = st.session_state["basicart_feat_value"]

        st.info(
            f"DURATA :: {feat['durata']:.1f}s  |  "
            f"FRAME :: {feat['n_frames']}  |  "
            f"SR :: {feat['sr']} Hz  |  "
            f"BPM stimato :: {feat['bpm']:.0f}"
        )

        # anteprima live al picco energetico del brano: si rigenera automaticamente
        # ad ogni modifica di forma/colori/slider, senza rifare l'analisi DSP
        with st.spinner("Aggiornamento anteprima :: Updating preview..."):
            anteprima_bgr, i_picco = genera_anteprima(
                feat, width, height, colore_bg, colore_fg, forma,
                densita, spessore, reattivita,
            )
        anteprima_rgb = cv2.cvtColor(anteprima_bgr, cv2.COLOR_BGR2RGB)
        st.image(
            anteprima_rgb,
            caption=f"Anteprima al picco audio ({i_picco / FPS:.1f}s) :: "
                    f"Preview at audio peak ({i_picco / FPS:.1f}s)",
            use_container_width=True,
        )

        n_step_stimato = MOTORI[forma]["n_step"] * densita
        fattore_forma = n_step_stimato / 900
        st.caption(
            "Tempo di render stimato :: Estimated render time ~ "
            f"{(width * height) / (1280 * 720) * 0.55 * fattore_forma:.0f}s ogni minuto di brano "
            "(indicativo, dipende dal server)"
        )
        if st.button("Genera video :: Generate video", type="primary"):
            path_in = st.session_state["basicart_audio_path"]
            with tempfile.TemporaryDirectory() as tmp:
                path_video_muto = os.path.join(tmp, "video_muto.mp4")
                genera_video(
                    feat, path_video_muto, width, height, colore_bg, colore_fg, forma,
                    densita=densita, spessore=spessore, reattivita=reattivita,
                )

                path_finale = os.path.join(tmp, "basicart_output.mp4")
                with st.spinner("Mixaggio audio/video :: Audio/video mixing..."):
                    mux_audio(path_video_muto, path_in, path_finale, feat["durata"])

                with open(path_finale, "rb") as f:
                    video_bytes = f.read()

                nome_file = "basicart_" + os.path.splitext(file_audio.name)[0] + ".mp4"
                st.session_state["basicart_video"] = video_bytes
                st.session_state["basicart_filename"] = nome_file
                report_md, report_txt = genera_report(
                    nome_file, forma, width, height, risoluzione_label, feat,
                    hex_bg, hex_fg, densita, spessore, reattivita, seed=507,
                )
                st.session_state["basicart_report_md"] = report_md
                st.session_state["basicart_report_txt"] = report_txt
                st.session_state["basicart_report_filename"] = (
                    os.path.splitext(nome_file)[0] + "_report.txt"
                )

    if "basicart_video" in st.session_state:
        st.success("RENDER COMPLETATO :: Render complete")
        st.video(st.session_state["basicart_video"])
        st.download_button(
            "Scarica video :: Download video",
            data=st.session_state["basicart_video"],
            file_name=st.session_state["basicart_filename"],
            mime="video/mp4",
        )
        if "basicart_report_md" in st.session_state:
            st.markdown(st.session_state["basicart_report_md"])
            st.download_button(
                "Scarica report :: Download report",
                data=st.session_state["basicart_report_txt"],
                file_name=st.session_state["basicart_report_filename"],
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
