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
- "Deriva"     : x=f(t2), y=f(t1)                  (coordinate cartesiane)
- "Fioritura"  : r=f(t2), a=f(t1), x=r*cos(a) ...  (coordinate polari)

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

FORME = ["Deriva (cartesiana)", "Fioritura (polare)", "Pulviscolo (cartesiana)", "Graffio (random walk)", "Sismografo (verticali)", "Frontiera (piano complesso)", "Aritmia (verticali)", "Iscrizione (testo a tempo)"]

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


def _spettro_a_barre(y, sr, hop_length, n_frames, n_barre=140):
    """Spettro a barre nel tempo: n_barre bande di frequenza log-spaziate
    (come un vero equalizzatore, piu' risoluzione sui bassi), energia
    normalizzata banda per banda sul proprio massimo nel brano. Usato per
    un display "spectrum analyzer" con barre ferme che pulsano in altezza,
    non per uno storico che scorre. Restituisce anche la classificazione
    di ogni barra (0=bassi <250Hz, 1=medi 250-2000Hz, 2=alti >=2000Hz) in
    base alla sua frequenza centrale, per poterla colorare di conseguenza."""
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)

    f_min, f_max = 40.0, min(sr / 2.0, 12000.0)
    bordi = np.geomspace(f_min, f_max, n_barre + 1)

    spettro = np.zeros((n_frames, n_barre))
    classe_banda = np.zeros(n_barre, dtype=np.int32)
    for b in range(n_barre):
        mask = (freqs >= bordi[b]) & (freqs < bordi[b + 1])
        if mask.any():
            colonna = S[mask].mean(axis=0)
        else:
            idx = int(np.argmin(np.abs(freqs - bordi[b])))
            colonna = S[idx]
        spettro[:, b] = _smussa(_norm(_adatta(colonna, n_frames)), alpha=0.30)

        centro = np.sqrt(bordi[b] * bordi[b + 1])
        if centro < 250:
            classe_banda[b] = 0
        elif centro < 2000:
            classe_banda[b] = 1
        else:
            classe_banda[b] = 2

    return spettro, classe_banda


def _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti):
    """Miscela i tre colori (bassi/medi/alti) pesandoli in base all'energia
    relativa di ciascuna banda nel brano in questo istante — usato dalle
    forme che non hanno gia' una struttura a bande (Deriva, Fioritura,
    Pulviscolo, Graffio, Frontiera), cosi' il colore stesso segue il
    timbro del brano momento per momento."""
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]
    tot = bassi + medi + alti + 1e-9
    wb, wm, wa = bassi / tot, medi / tot, alti / tot
    return tuple(wb * colore_bassi[c] + wm * colore_medi[c] + wa * colore_alti[c] for c in range(3))


def _stima_bpm_e_battiti(y, sr, hop_length, fps, n_frames):
    """Stima il tempo (BPM) del brano via beat tracking DSP (no AI) e
    restituisce anche l'indice-frame-video di ogni battito rilevato.
    Il BPM pilota la velocita' di rotazione/pulsazione del pattern
    (reinterpretazione originale, non presente negli esempi BASIC di
    riferimento); i battiti esatti servono per sincronizzare la
    rivelazione del testo in "Iscrizione" a tempo di musica reale,
    non a un intervallo fisso."""
    try:
        tempo, beat_hop = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        bpm = float(np.atleast_1d(tempo)[0])
        beat_frames_video = np.clip(
            (np.asarray(beat_hop) * hop_length / sr * fps).astype(np.int32), 0, n_frames - 1
        )
    except Exception:
        bpm = 120.0
        beat_frames_video = np.array([], dtype=np.int32)
    if bpm <= 0:
        bpm = 120.0
    return bpm, beat_frames_video


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

    rms_grezzo = _adatta(rms, n_frames)   # RMS non normalizzato: serve per rilevare il
                                            # silenzio reale, non solo il punto piu' quieto
                                            # del brano (che con _norm risulterebbe sempre 0)
    rms = _norm(rms_grezzo)
    centroid = _norm(_adatta(centroid, n_frames))
    onset_env = _norm(_adatta(onset_env, n_frames))

    bassi, medi, alti = _bande_spettrali(y, sr, hop_length, n_frames)
    spettro, spettro_classe = _spettro_a_barre(y, sr, hop_length, n_frames)
    bpm, battiti_video = _stima_bpm_e_battiti(y, sr, hop_length, fps, n_frames)

    # gate di presenza: 0 nel silenzio vero, 1 appena il volume supera una soglia
    # relativa al picco del brano — usato per far scomparire/fermare l'animazione
    # nei momenti silenziosi invece di limitarsi a rimpicciolirla
    picco = rms_grezzo.max()
    if picco > 1e-9:
        presenza = np.clip(rms_grezzo / (picco * 0.10), 0.0, 1.0)
    else:
        presenza = np.zeros(n_frames)
    presenza = _smussa(presenza, alpha=0.15)

    return {
        "rms": _smussa(rms),
        "centroid": _smussa(centroid),
        "onset": _smussa(onset_env, alpha=0.20),   # onset resta un po' piu' reattivo
        "bassi": _smussa(bassi),
        "medi": _smussa(medi),
        "alti": _smussa(alti),
        "spettro": spettro,
        "spettro_classe": spettro_classe,
        "presenza": presenza,
        "bpm": bpm,
        "battiti_video": battiti_video,
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
    presenza = feat["presenza"][i]   # 0 nel silenzio vero, 1 a volume normale

    energia = (0.6 * rms + 0.4 * bassi) * reattivita

    # velocita' di rotazione/pulsazione ancorata al BPM del brano (reinterpretazione
    # originale: un brano a 160 BPM "gira" piu' veloce di una ballata a 70 BPM)
    velocita = np.clip(bpm / 120.0, 0.55, 1.9)

    # pulsazione "a respiro" sincronizzata al beat: una lieve oscillazione
    # dell'ampiezza al ritmo esatto del brano, indipendente dalla reattivita'
    # istantanea dell'onset — tocco creativo, non presente nei riferimenti BASIC
    fase_beat = 2 * np.pi * (bpm / 60.0) * (t_frame / fps)
    respiro = 1.0 + 0.06 * np.sin(fase_beat)

    # il gate di presenza azzera quasi del tutto ampiezza e luminosita' nel
    # silenzio: l'animazione si raccoglie a un punto e sfuma via con la scia,
    # invece di continuare a disegnare a meta' intensita' come prima
    fattore_ampiezza = np.clip((0.15 + 0.85 * energia) * respiro, 0.03, 2.0) * presenza

    k1 = np.clip(2.0 + (3.0 * centroid + 2.0 * alti) * reattivita, 2.0, 9.0)   # timbro/alti
    k2 = np.clip(2.0 + 3.0 * medi * reattivita, 2.0, 9.0)                       # medi

    # per il pattern Fioritura (polare) le due componenti r e a devono condividere
    # la STESSA frequenza secondaria (come nel riferimento BASIC originale,
    # k=3.5 fisso per entrambe): usare due k diversi rompe la correlazione
    # armonica e trasforma i petali puliti in un groviglio denso stile Lissajous
    k_loto = np.clip(2.6 + (1.2 * centroid + 0.8 * alti + 0.6 * medi) * reattivita, 2.2, 5.2)

    intensita = np.clip(0.88 + 0.12 * onset, 0.5, 1.0) * presenza   # baseline alta a volume
                                                                       # normale, ma azzerata
                                                                       # dal gate nel silenzio
    return fattore_ampiezza, k1, k2, k_loto, onset, intensita, velocita


def disegna_ellisse(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
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

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    colore = tuple(min(int(c * intensita), 255) for c in colore_base)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=colore, thickness=spessore, lineType=cv2.LINE_AA)
    return canvas


def disegna_loto(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                  colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """Pattern polare: r=f(t2), a=f(t1), x=r*cos(a), y=r*sin(a).
    r e a condividono la stessa frequenza secondaria k (come nel BASIC
    originale) per mantenere la simmetria a petali. A differenza
    dell'Deriva, qui i punti vengono tracciati SPARSI (stile PLOT del
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

    colore = np.array([min(int(c * intensita), 255) for c in _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)], dtype=np.uint8)

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


def disegna_pulviscolo(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                        colore_alti, t1_arr, fps, reattivita=1.0, spessore=1, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """Struttura radicalmente diversa dalle altre due: una spirale di
    polvere che si espande dal centro verso il bordo (non ellissi chiuse
    ne' petali), pilotata dall'audio. Il numero di giri della spirale
    dipende dal timbro/alti del brano, la rotazione complessiva dal BPM.
    Aggiunge un tocco mio non presente in alcun riferimento: un lieve
    rumore stocastico radiale proporzionale alle alte frequenze, che da'
    una texture "polverosa" — la grana si addensa quando il brano e'
    brillante/ricco di alti, si dirada quando e' piu' cupo."""
    fattore, k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    alti = feat["alti"][i] * reattivita

    n = len(t1_arr)
    frac = t1_arr / n   # 0..1 lungo il braccio della spirale, dal centro al bordo

    # numero di giri della spirale: pilotato da timbro/alti (k1 gia' li combina)
    giri = 1.5 + 3.0 * np.clip((k1 - 2.0) / 7.0, 0.0, 1.0)
    theta = frac * giri * 2 * np.pi

    rotazione = t_frame * 0.012 * velocita + onset * 0.5   # l'intera spirale ruota nel tempo
    raggio_punto = frac * fattore

    # granulosita' stocastica radiale legata agli alti: tocco creativo mio,
    # non presente nel riferimento BASIC originale
    grana = 0.05 * alti
    if grana > 0.002:
        raggio_punto = raggio_punto + np.random.uniform(-grana, grana, size=raggio_punto.shape)

    xs = (cx + raggio_x * raggio_punto * np.cos(theta + rotazione)).astype(np.int32)
    ys = (cy + raggio_y * raggio_punto * np.sin(theta + rotazione)).astype(np.int32)

    h, w = canvas.shape[:2]
    dentro = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[dentro], ys[dentro]

    colore = np.array([min(int(c * intensita), 255) for c in _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)], dtype=np.uint8)
    if spessore <= 1:
        canvas[ys, xs] = colore
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ys, xs] = 255
        kernel = np.ones((spessore, spessore), np.uint8)
        mask = cv2.dilate(mask, kernel)
        canvas[mask > 0] = colore

    return canvas


def disegna_graffio(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """Random walk di segmenti brevi che vaga per il fotogramma con
    teletrasporto ai bordi (wrap-around) — ispirato a un terzo riferimento
    BASIC che usa RANDOM invece di funzioni trigonometriche. A differenza
    del riferimento (colore casuale per ogni linea), qui il colore resta
    monocromatico e coerente col resto dell'app. Il passo del vagabondaggio
    e' pilotato da energia/bassi (fattore_ampiezza, che incorpora gia' il
    gate di silenzio: nei momenti quieti il tratto quasi si ferma), la
    lunghezza dei segmenti dagli alti. Usa np.cumsum per vettorizzare la
    sequenza di passi casuali invece di un loop Python punto-per-punto."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, _velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    alti = feat["alti"][i] * reattivita

    h, w = canvas.shape[:2]
    if stato is None or "x" not in stato:
        if stato is None:
            stato = {}
        stato["x"] = w / 2.0
        stato["y"] = h / 2.0

    n_step = len(t1_arr)   # la densita' (slider) regola quanti segmenti per frame

    passo_max = 2.0 + 14.0 * fattore       # energia/bassi + gate silenzio gia' inclusi
    lunghezza_max = 2.0 + 30.0 * alti      # timbro/alti -> segmenti piu' o meno lunghi

    passi_x = np.random.uniform(-passo_max, passo_max, size=n_step)
    passi_y = np.random.uniform(-passo_max, passo_max, size=n_step)

    xs = np.mod(stato["x"] + np.cumsum(passi_x), w)
    ys = np.mod(stato["y"] + np.cumsum(passi_y), h)

    dx = np.random.uniform(-lunghezza_max, lunghezza_max, size=n_step)
    dy = np.random.uniform(-lunghezza_max, lunghezza_max, size=n_step)

    colore = tuple(min(int(c * intensita), 255) for c in _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti))

    for k in range(n_step):
        x1, y1 = int(xs[k]), int(ys[k])
        x2, y2 = int(xs[k] + dx[k]), int(ys[k] + dy[k])
        cv2.line(canvas, (x1, y1), (x2, y2), colore, spessore, lineType=cv2.LINE_AA)

    stato["x"], stato["y"] = float(xs[-1]), float(ys[-1])

    return canvas


def disegna_sismografo(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                        colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """Spectrum analyzer: barre verticali FERME in posizione orizzontale
    (non scorrono lateralmente) — ogni barra rappresenta una banda di
    frequenza log-spaziata (come un equalizzatore reale, precalcolata una
    sola volta su tutto il brano) e la sua altezza pulsa su e giu' nel
    tempo seguendo l'energia di quella banda. Estensione simmetrica sopra/
    sotto la linea centrale. Ogni barra usa il colore (bassi/medi/alti)
    corrispondente alla sua frequenza — non un'unica tinta piatta.
    Luminosita' e spessore variano leggermente barra per barra. Beneficia
    del gate di silenzio (spettro azzerato nel silenzio vero). Il numero
    di barre e' fisso (non controllato dallo slider densita', perche' lo
    spettro e' precalcolato una sola volta in analisi)."""
    _fattore, _k1, _k2, _k_loto, _onset, intensita, _velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    presenza = feat["presenza"][i]

    h, w = canvas.shape[:2]
    spettro = np.clip(feat["spettro"][i] * reattivita * presenza, 0.0, 1.6)
    classe = feat["spettro_classe"]
    n_barre = len(spettro)
    palette = [colore_bassi, colore_medi, colore_alti]

    spacing = w / n_barre
    cy_int = h // 2
    altezze = (spettro * (h * 0.46)).astype(np.int32)
    spessori_random = np.random.uniform(0.75, 1.25, size=n_barre)

    for k in range(n_barre):
        x = int(k * spacing + spacing / 2)
        alt = altezze[k]

        # luminosita' proporzionale all'altezza: le barre piu' energiche
        # brillano di piu', quelle basse restano tenui — tocco artistico
        frazione = np.clip(spettro[k], 0.0, 1.0)
        intens_barra = intensita * (0.40 + 0.60 * frazione)
        colore_banda = palette[classe[k]]
        colore = tuple(min(int(c * intens_barra), 255) for c in colore_banda)
        spess = max(1, int(round(spessore * spessori_random[k])))

        cv2.line(canvas, (x, cy_int - alt), (x, cy_int + alt), colore, spess, lineType=cv2.LINE_AA)

    return canvas


def disegna_julia(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                   colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """Insieme di Julia (frattale nel piano complesso: z=z^2+c iterato per
    ogni punto), animato facendo ruotare la costante c nel tempo in base
    all'audio — piccole variazioni di c producono forme del frattale
    drasticamente diverse, quindi un brano potente fa "vibrare" la forma
    in modo brusco. Reinterpretazione libera: l'algoritmo di Julia e'
    matematica standard, non copiato dal riferimento BASIC (che disegnava
    il frattale in modo statico). Calcolato a risoluzione ridotta e
    ingrandito con interpolazione nearest per un effetto deliberatamente
    "a blocchi", coerente con l'estetica BASIC primitivo, e per restare
    performante (l'alternativa, calcolo pixel-per-pixel a piena risoluzione
    per ogni frame, sarebbe troppo lenta per un video)."""
    fattore, k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    alti = feat["alti"][i] * reattivita

    h, w = canvas.shape[:2]

    # risoluzione di calcolo ridotta (la densita' la scala tramite n_step)
    ris_w = max(60, int(np.sqrt(len(t1_arr)) * 8))
    ris_h = max(40, int(ris_w * h / w))

    # la costante c ruota nel tempo: raggio pilotato da energia/bassi (fattore
    # incorpora gia' il gate di silenzio), velocita' angolare dal BPM, piccoli
    # scatti sugli attacchi (onset)
    raggio_c = 0.55 + 0.25 * np.clip(fattore, 0.0, 1.5)
    angolo_c = t_frame * 0.02 * velocita + onset * 0.6 + k1 * 0.05
    c = complex(raggio_c * np.cos(angolo_c), raggio_c * np.sin(angolo_c))

    xs_lin = np.linspace(-1.5, 1.5, ris_w)
    ys_lin = np.linspace(-1.5 * ris_h / ris_w, 1.5 * ris_h / ris_w, ris_h)
    X, Y = np.meshgrid(xs_lin, ys_lin)
    Z = X + 1j * Y

    n_iter = 28
    escape = np.zeros((ris_h, ris_w), dtype=np.float32)
    attivo = np.ones((ris_h, ris_w), dtype=bool)
    for it in range(n_iter):
        Z[attivo] = Z[attivo] ** 2 + c
        scappati_ora = attivo & (np.abs(Z) > 2.0)
        escape[scappati_ora] = it
        attivo &= ~scappati_ora
        if not attivo.any():
            break
    # i punti che non scappano mai (interno dell'insieme) restano a 0 (sfondo):
    # solo il bordo/l'esterno (dove l'escape time varia) produce la texture

    valore = np.clip((escape / n_iter) * (0.6 + 0.4 * alti), 0.0, 1.0)
    campo_u8 = (valore * 255).astype(np.uint8)
    campo_grande = cv2.resize(campo_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    colore_arr = np.array(_colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti), dtype=np.float32)
    campo_col = (campo_grande[..., None].astype(np.float32) / 255.0) * colore_arr * intensita
    canvas[:] = np.clip(campo_col, 0, 255).astype(np.uint8)

    return canvas


def disegna_aritmia(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """Variante di Sismografo: STESSO motore (spettro di frequenza fisso,
    140 barre in posizione orizzontale FERMA, nessuno storico che scorre),
    ma ogni barra ha una direzione (su o giu') decisa UNA SOLA VOLTA
    all'inizio del render e mai piu' cambiata — non scorrevole, solo
    stilisticamente diversa: invece di estendersi simmetricamente sopra e
    sotto come Sismografo, ogni barra fissa si estende in una sola
    direzione, dando un profilo a gradini irregolare (alcune barre su,
    altre giu') pur restando ancorate alla loro posizione di frequenza.
    Ogni barra usa il colore (bassi/medi/alti) della sua banda."""
    _fattore, _k1, _k2, _k_loto, _onset, intensita, _velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    presenza = feat["presenza"][i]

    h, w = canvas.shape[:2]
    spettro = np.clip(feat["spettro"][i] * reattivita * presenza, 0.0, 1.6)
    classe = feat["spettro_classe"]
    n_barre = len(spettro)
    palette = [colore_bassi, colore_medi, colore_alti]

    if stato is None:
        stato = {}
    if "direzioni" not in stato or len(stato["direzioni"]) != n_barre:
        # direzione fissata UNA VOLTA SOLA per l'intero render, non ad ogni
        # frame: non e' un meccanismo scorrevole, e' una scelta statica
        stato["direzioni"] = np.random.choice([-1.0, 1.0], size=n_barre)
    direzioni = stato["direzioni"]

    spacing = w / n_barre
    cy_int = h // 2
    altezze = (spettro * (h * 0.46)).astype(np.int32)
    spessori_random = np.random.uniform(0.75, 1.25, size=n_barre)

    for k in range(n_barre):
        x = int(k * spacing + spacing / 2)
        y2 = cy_int + int(altezze[k] * direzioni[k])

        frazione = np.clip(spettro[k], 0.0, 1.0)
        intens_barra = intensita * (0.40 + 0.60 * frazione)
        colore_banda = palette[classe[k]]
        colore = tuple(min(int(c * intens_barra), 255) for c in colore_banda)
        spess = max(1, int(round(spessore * spessori_random[k])))

        cv2.line(canvas, (x, cy_int), (x, y2), colore, spess, lineType=cv2.LINE_AA)

    return canvas


_ALFABETO_ISCRIZIONE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_FONT_ISCRIZIONE = {
    0: cv2.FONT_HERSHEY_SIMPLEX,
    1: cv2.FONT_HERSHEY_DUPLEX,
    2: cv2.FONT_HERSHEY_TRIPLEX,
    3: cv2.FONT_HERSHEY_COMPLEX,
    4: cv2.FONT_HERSHEY_PLAIN,
}


def disegna_iscrizione(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                        colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                        dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
    """La frase si materializza da uno sciame di lettere casuali che vagano
    per lo schermo — stile "decrittazione": ogni lettera del bersaglio vaga
    liberamente mostrando caratteri casuali finche' non arriva il suo turno,
    poi scivola dolcemente nella posizione finale e si blocca sulla lettera
    corretta. Lettere decorative extra continuano a vagare sullo sfondo
    senza mai bloccarsi, per riempire lo schermo. I tempi di blocco sono
    calcolati per completare SEMPRE la frase entro la fine del brano: se ci
    sono abbastanza battiti rilevati li usa (un carattere per battito),
    altrimenti distribuisce i caratteri in modo uniforme su tutta la durata
    — cosi' anche un brano breve con una frase corta arriva sempre a
    completarsi. Reinterpretazione libera (non copia) della texture a
    caratteri di un pattern C64 BASIC V2 mostrato come riferimento."""
    if not frase:
        return canvas

    h, w = canvas.shape[:2]
    font = _FONT_ISCRIZIONE.get(font_scelto, cv2.FONT_HERSHEY_SIMPLEX)
    colore = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    _fattore, _k1, _k2, _k_loto, _onset, intensita, _velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    n_car = len(frase)
    n_frames_tot = feat["n_frames"]

    if stato is None:
        stato = {}

    if "isc_lock_frames" not in stato:
        # dimensione e posizione bersaglio di ogni carattere, calcolate una
        # sola volta: la frase intera determina la scala (non cambia mentre
        # le lettere si assestano)
        scala = 2.0 * dimensione_testo
        spessore_testo = max(2, spessore + 1)
        larghezza_target = raggio_x * 1.9
        (tw_f, th_f), _ = cv2.getTextSize(frase, font, scala, spessore_testo)
        if tw_f > 0:
            scala *= min(1.0, larghezza_target / tw_f)
        (tw_f, th_f), _ = cv2.getTextSize(frase, font, scala, spessore_testo)
        x0 = cx - tw_f / 2
        y0 = cy + th_f / 2

        posizioni_target = []
        for k in range(n_car):
            (w_prefisso, _), _ = cv2.getTextSize(frase[:k], font, scala, spessore_testo)
            posizioni_target.append(np.array([x0 + w_prefisso, y0]))

        # tempi di blocco: un carattere per battito se ce ne sono abbastanza,
        # altrimenti distribuiti uniformemente su tutta la durata del brano
        battiti = feat["battiti_video"]
        lock_frames = np.zeros(n_car, dtype=np.int32)
        if len(battiti) >= n_car:
            lock_frames[:] = battiti[:n_car]
        else:
            for k in range(n_car):
                lock_frames[k] = int(round((k + 1) / n_car * (n_frames_tot - 1)))

        rng = np.random.default_rng(507)
        margine = 40
        pos_correnti = np.array([
            [rng.uniform(margine, max(margine + 1, w - margine)),
             rng.uniform(margine, max(margine + 1, h - margine))]
            for _ in range(n_car)
        ])
        vel = rng.uniform(-1.6, 1.6, size=(n_car, 2))

        n_extra = max(0, int(lettere_extra))
        pos_extra = np.array([[rng.uniform(0, w), rng.uniform(0, h)] for _ in range(n_extra)]) \
            if n_extra > 0 else np.zeros((0, 2))
        vel_extra = rng.uniform(-1.3, 1.3, size=(n_extra, 2)) if n_extra > 0 else np.zeros((0, 2))

        stato["isc_font"] = font
        stato["isc_scala"] = scala
        stato["isc_spessore_testo"] = spessore_testo
        stato["isc_pos_target"] = posizioni_target
        stato["isc_lock_frames"] = lock_frames
        stato["isc_pos"] = pos_correnti
        stato["isc_vel"] = vel
        stato["isc_bloccato"] = np.zeros(n_car, dtype=bool)
        stato["isc_lettera"] = [rng.choice(list(_ALFABETO_ISCRIZIONE)) for _ in range(n_car)]
        stato["isc_pos_extra"] = pos_extra
        stato["isc_vel_extra"] = vel_extra
        stato["isc_lettera_extra"] = [rng.choice(list(_ALFABETO_ISCRIZIONE)) for _ in range(n_extra)]
        stato["isc_rng"] = rng

        # se l'anteprima parte da un frame gia' avanzato nel brano, i
        # caratteri il cui turno e' gia' passato vanno mostrati subito
        # bloccati, non fatti ripartire dal vagabondaggio
        for k in range(n_car):
            if lock_frames[k] <= i:
                stato["isc_bloccato"][k] = True
                stato["isc_pos"][k] = posizioni_target[k]
                stato["isc_lettera"][k] = frase[k]

    font = stato["isc_font"]
    scala = stato["isc_scala"]
    spessore_testo = stato["isc_spessore_testo"]
    rng = stato["isc_rng"]
    transizione = max(1, int(0.6 * fps))
    margine = 20

    def rimbalza(pos, vel):
        for ax, lim in ((0, w), (1, h)):
            if pos[ax] < margine or pos[ax] > lim - margine:
                vel[ax] *= -1

    for k in range(n_car):
        if stato["isc_bloccato"][k]:
            continue
        lf = stato["isc_lock_frames"][k]
        if i >= lf:
            stato["isc_bloccato"][k] = True
            stato["isc_pos"][k] = stato["isc_pos_target"][k]
            stato["isc_lettera"][k] = frase[k]
        elif i >= lf - transizione:
            t_rel = (i - (lf - transizione)) / transizione
            stato["isc_pos"][k] = (1 - t_rel) * stato["isc_pos"][k] + t_rel * stato["isc_pos_target"][k]
            if i % 3 == 0:
                stato["isc_lettera"][k] = rng.choice(list(_ALFABETO_ISCRIZIONE))
        else:
            rimbalza(stato["isc_pos"][k], stato["isc_vel"][k])
            stato["isc_pos"][k] = stato["isc_pos"][k] + stato["isc_vel"][k]
            if i % 4 == 0:
                stato["isc_lettera"][k] = rng.choice(list(_ALFABETO_ISCRIZIONE))

    for j in range(len(stato["isc_pos_extra"])):
        rimbalza(stato["isc_pos_extra"][j], stato["isc_vel_extra"][j])
        stato["isc_pos_extra"][j] = stato["isc_pos_extra"][j] + stato["isc_vel_extra"][j]
        if i % 5 == 0:
            stato["isc_lettera_extra"][j] = rng.choice(list(_ALFABETO_ISCRIZIONE))

    colore_extra = tuple(min(int(c * intensita * 0.35), 255) for c in colore)
    for j in range(len(stato["isc_pos_extra"])):
        x, y = stato["isc_pos_extra"][j]
        cv2.putText(canvas, stato["isc_lettera_extra"][j], (int(x), int(y)), font,
                    scala * 0.55, colore_extra, max(1, spessore_testo - 1), cv2.LINE_AA)

    colore_int = tuple(min(int(c * intensita), 255) for c in colore)
    for k in range(n_car):
        x, y = stato["isc_pos"][k]
        cv2.putText(canvas, stato["isc_lettera"][k], (int(x), int(y)), font,
                    scala, colore_int, spessore_testo, cv2.LINE_AA)

    return canvas


MOTORI = {
    "Deriva (cartesiana)": {"funzione": disegna_ellisse, "n_step": 900, "fade": 0.90},
    "Fioritura (polare)": {"funzione": disegna_loto, "n_step": 3300, "fade": 0.80},
    "Pulviscolo (cartesiana)": {"funzione": disegna_pulviscolo, "n_step": 2500, "fade": 0.88},
    "Graffio (random walk)": {"funzione": disegna_graffio, "n_step": 60, "fade": 0.85},
    "Sismografo (verticali)": {"funzione": disegna_sismografo, "n_step": 160, "fade": 0.0},
    "Frontiera (piano complesso)": {"funzione": disegna_julia, "n_step": 900, "fade": 0.0},
    "Aritmia (verticali)": {"funzione": disegna_aritmia, "n_step": 160, "fade": 0.0},
    "Iscrizione (testo a tempo)": {"funzione": disegna_iscrizione, "n_step": 100, "fade": 0.0},
}


def genera_video(feat, path_out, width, height, colore_bg, colore_bassi, colore_medi, colore_alti,
                  forma, fps=FPS, seed=507, densita=1.0, spessore=2, reattivita=1.0, frase="",
                  dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
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
    stato = {}   # stato persistente (usato solo da Graffio/Aritmia/Iscrizione, ignorato dalle altre)

    for i in range(n_frames):
        # fade verso il nero (scia stile fosfori)
        canvas = canvas * fade_alpha + bg * (1 - fade_alpha)

        canvas_u8 = canvas.astype(np.uint8)
        canvas_u8 = disegna(
            canvas_u8, t_frame=i, feat=feat, i=i,
            cx=cx, cy=cy, raggio_x=raggio_x, raggio_y=raggio_y,
            colore_bassi=colore_bassi, colore_medi=colore_medi, colore_alti=colore_alti,
            t1_arr=t1_arr, fps=fps,
            reattivita=reattivita, spessore=spessore, stato=stato, frase=frase,
            dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra,
        )
        canvas = canvas_u8.astype(np.float32)

        writer.write(canvas_u8)

        if i % 5 == 0 or i == n_frames - 1:
            progress.progress((i + 1) / n_frames, text=f"RENDER :: frame {i+1}/{n_frames}")

    writer.release()
    progress.empty()


def genera_anteprima(feat, width, height, colore_bg, colore_bassi, colore_medi, colore_alti, forma,
                      densita, spessore, reattivita, seed=507, finestra_s=4.0, fps=FPS, frase="",
                      dimensione_testo=1.0, font_scelto=0, lettere_extra=40):
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
    stato = {}   # stato persistente (usato solo da Graffio/Aritmia)

    for i in range(i_inizio, i_picco + 1):
        canvas = canvas * fade_alpha + bg * (1 - fade_alpha)
        canvas_u8 = canvas.astype(np.uint8)
        canvas_u8 = disegna(
            canvas_u8, t_frame=i, feat=feat, i=i,
            cx=cx, cy=cy, raggio_x=raggio_x, raggio_y=raggio_y,
            colore_bassi=colore_bassi, colore_medi=colore_medi, colore_alti=colore_alti,
            t1_arr=t1_arr, fps=fps,
            reattivita=reattivita, spessore=spessore, stato=stato, frase=frase,
            dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra,
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
                   hex_bg, hex_bassi, hex_medi, hex_alti, densita, spessore, reattivita, seed, vol):
    """Report bilingue IT/EN stile Loop507 (blocco IT completo seguito dal
    blocco EN completo, formato compatto senza separatori — come da
    modello fornito). Restituisce sia la versione da mostrare in chat
    (con code-fence) sia il testo grezzo per il download."""
    it = (
        f":: BASICART // Vol. {vol:03d}\n"
        "REPORT ::\n"
        f"FILE            :: {nome_file}\n"
        f"FORMA           :: {forma}\n"
        f"FORMATO         :: {risoluzione_label}\n"
        f"DURATA          :: {feat['durata']:.1f}s\n"
        f"FRAME           :: {feat['n_frames']} @ {FPS}fps\n"
        f"SAMPLE RATE     :: {feat['sr']} Hz\n"
        f"BPM STIMATO     :: {feat['bpm']:.0f}\n"
        f"COLORE SFONDO   :: {hex_bg}\n"
        f"COLORE BASSI    :: {hex_bassi}\n"
        f"COLORE MEDI     :: {hex_medi}\n"
        f"COLORE ALTI     :: {hex_alti}\n"
        f"DENSITA'        :: {densita:.2f}x\n"
        f"SPESSORE        :: {spessore}\n"
        f"REATTIVITA'     :: {reattivita:.2f}x\n"
        f"SEED            :: {seed}\n"
        "DSP :: RMS, spectral centroid, onset strength, bande bassi/medi/\n"
        "       alti (STFT), beat tracking — nessun modello AI/neurale.\n"
        "Regia e Algoritmo :: Loop507\n"
        "L'audio e' stato scomposto in frequenze. Il codice ne ha ridisegnato la forma."
    )

    en = (
        f":: BASICART // Vol. {vol:03d}\n"
        "REPORT ::\n"
        f"FILE            :: {nome_file}\n"
        f"SHAPE           :: {forma}\n"
        f"RESOLUTION      :: {risoluzione_label}\n"
        f"DURATION        :: {feat['durata']:.1f}s\n"
        f"FRAMES          :: {feat['n_frames']} @ {FPS}fps\n"
        f"SAMPLE RATE     :: {feat['sr']} Hz\n"
        f"ESTIMATED BPM   :: {feat['bpm']:.0f}\n"
        f"BACKGROUND COLOR:: {hex_bg}\n"
        f"LOW COLOR       :: {hex_bassi}\n"
        f"MID COLOR       :: {hex_medi}\n"
        f"HIGH COLOR      :: {hex_alti}\n"
        f"DENSITY         :: {densita:.2f}x\n"
        f"THICKNESS       :: {spessore}\n"
        f"REACTIVITY      :: {reattivita:.2f}x\n"
        f"SEED            :: {seed}\n"
        "DSP :: RMS, spectral centroid, onset strength, low/mid/high\n"
        "       bands (STFT), beat tracking — no AI/neural model.\n"
        "Direction & Algorithm :: Loop507\n"
        "The audio was broken down into frequencies. The code redrew its shape."
    )

    tag_forma = forma.split(" ")[0]   # "Deriva" o "Fioritura"
    hashtags = (
        f"#BasicArt #{tag_forma} #GenerativeArt #AudioReactive "
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

    frase = ""
    dimensione_testo = 1.0
    font_scelto = 0
    lettere_extra = 40
    if forma == "Iscrizione (testo a tempo)":
        frase = st.text_input(
            "Frase da scrivere a tempo :: Phrase to write in time",
            max_chars=60,
            help="Le lettere si assestano nella posizione giusta seguendo i battiti del "
                 "brano, completando sempre la frase entro la fine :: Letters settle into "
                 "place following the track's beats, always completing by the end"
        ).upper()

        st.caption("Controlli dedicati :: Dedicated controls — Iscrizione")
        colA, colB, colC = st.columns(3)
        with colA:
            dimensione_testo = st.slider(
                "Dimensione testo :: Text size", 0.5, 2.0, 1.0, 0.1
            )
        with colB:
            nomi_font = {
                0: "Semplice :: Simple", 1: "Doppio :: Double", 2: "Triplo :: Triple",
                3: "Complesso :: Complex", 4: "Stampatello :: Plain",
            }
            font_scelto = st.selectbox(
                "Font", options=list(nomi_font.keys()), format_func=lambda k: nomi_font[k]
            )
        with colC:
            lettere_extra = st.slider(
                "Lettere fluttuanti :: Floating letters", 0, 120, 40, 5,
                help="Quante lettere decorative vagano sullo sfondo senza mai assestarsi :: "
                     "How many decorative letters wander in the background without settling"
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
        st.caption("Colori animazione per banda di frequenza :: Animation colors per frequency band")

    col5, col6, col7 = st.columns(3)
    with col5:
        hex_bassi = st.color_picker("Bassi :: Low", "#ff5050")
    with col6:
        hex_medi = st.color_picker("Medi :: Mid", "#ffffff")
    with col7:
        hex_alti = st.color_picker("Alti :: High", "#50a0ff")

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
    colore_bassi = hex_a_bgr(hex_bassi)
    colore_medi = hex_a_bgr(hex_medi)
    colore_alti = hex_a_bgr(hex_alti)
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
                feat, width, height, colore_bg, colore_bassi, colore_medi, colore_alti, forma,
                densita, spessore, reattivita, frase=frase,
                dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra,
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
                    feat, path_video_muto, width, height, colore_bg,
                    colore_bassi, colore_medi, colore_alti, forma,
                    densita=densita, spessore=spessore, reattivita=reattivita, frase=frase,
                    dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra,
                )

                path_finale = os.path.join(tmp, "basicart_output.mp4")
                with st.spinner("Mixaggio audio/video :: Audio/video mixing..."):
                    mux_audio(path_video_muto, path_in, path_finale, feat["durata"])

                with open(path_finale, "rb") as f:
                    video_bytes = f.read()

                nome_file = "basicart_" + os.path.splitext(file_audio.name)[0] + ".mp4"
                st.session_state["basicart_video"] = video_bytes
                st.session_state["basicart_filename"] = nome_file

                # numero di volume progressivo (catalogo Loop507): parte da 000
                # e sale di uno ad ogni video generato in questa sessione
                vol = st.session_state.get("basicart_vol_counter", -1) + 1
                st.session_state["basicart_vol_counter"] = vol

                report_md, report_txt = genera_report(
                    nome_file, forma, width, height, risoluzione_label, feat,
                    hex_bg, hex_bassi, hex_medi, hex_alti, densita, spessore, reattivita,
                    seed=507, vol=vol,
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
            st.subheader("Report")
            st.markdown(st.session_state["basicart_report_md"])
            st.download_button(
                "Scarica report :: Download report",
                data=st.session_state["basicart_report_txt"],
                file_name=st.session_state["basicart_report_filename"],
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
