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
import functools
import tempfile
import numpy as np
import cv2
import streamlit as st
import librosa
from moviepy import VideoFileClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

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

FORME = ["Deriva (cartesiana)", "Fioritura (polare)", "Pulviscolo (cartesiana)", "Graffio (random walk)", "Sismografo (verticali)", "Frontiera (piano complesso)", "Aritmia (verticali)", "Iscrizione (testo a tempo)", "Sinapsi (rete)", "Labirinto (tasselli)", "Risonanza (placca)", "Statica (automa)", "Poliedro (wireframe)", "Epicicli (Fourier)", "Plasma (interferenza)", "Cometa (starfield prospettico)", "Magma (metaballs)", "Mosaico (celle di Voronoi)", "Galleria (tunnel prospettico)", "Braci (fuoco algoritmico)", "Vita (automa cellulare 2D)"]

# font veri (TTF) per "Iscrizione" — cartella "fonts/" accanto a questo script.
# Se mancante, l'app ripiega automaticamente sui font Hershey di OpenCV
# (piu' grezzi ma sempre disponibili, nessuna dipendenza esterna)
_CARTELLA_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_FONT_TTF = {
    0: "GeistMono-Bold.ttf",
    1: "BigShoulders-Bold.ttf",
    2: "IBMPlexSerif-Bold.ttf",
    3: "Boldonse-Regular.ttf",
    4: "EricaOne-Regular.ttf",
}
_FONT_TTF_NOMI = {
    0: "Geist Mono (tecnico)", 1: "Big Shoulders (bold display)",
    2: "IBM Plex Serif (elegante)", 3: "Boldonse (grafico/pesante)",
    4: "Erica One (rotondo)",
}


@functools.lru_cache(maxsize=32)
def _carica_font_ttf(font_scelto, dimensione_px):
    """Carica un font TTF alla dimensione richiesta, con cache (evita di
    ricaricare il file da disco ad ogni frame). Torna None se il file non
    e' disponibile, cosi' il chiamante puo' ripiegare su OpenCV/Hershey."""
    nome_file = _FONT_TTF.get(font_scelto, _FONT_TTF[0])
    percorso = os.path.join(_CARTELLA_FONT, nome_file)
    try:
        return ImageFont.truetype(percorso, max(8, int(dimensione_px)))
    except Exception:
        return None

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

    bassi_raw = _adatta(energia(m_bassi), n_frames)
    medi_raw = _adatta(energia(m_medi), n_frames)
    alti_raw = _adatta(energia(m_alti), n_frames)

    # quote di dominanza relativa (per il MIX colore): calcolate sui valori
    # GREZZI (non normalizzati singolarmente) cosi' la somma dei tre pesi fa
    # sempre 1 e riflette davvero quale banda domina in quell'istante. Usare
    # invece bassi/medi/alti normalizzati ciascuno sul proprio range (come
    # prima) e' un bug: ogni banda viene stirata a riempire 0..1 sul proprio
    # massimo, quindi anche una banda quasi sempre debole (es. gli alti su un
    # brano cupo) risulta "alla pari" delle altre nel calcolo dei pesi, e il
    # colore miscelato resta quasi costante per tutto il brano invece di
    # spostarsi visibilmente verso bassi/medi/alti quando quella banda prevale
    tot_raw = bassi_raw + medi_raw + alti_raw + 1e-9
    quota_bassi = bassi_raw / tot_raw
    quota_medi = medi_raw / tot_raw
    quota_alti = alti_raw / tot_raw

    return (_norm(bassi_raw), _norm(medi_raw), _norm(alti_raw),
            quota_bassi, quota_medi, quota_alti)


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
    timbro del brano momento per momento. Usa le "quote" di dominanza
    (feat['quota_*'], calcolate su energia grezza, non normalizzata banda
    per banda) e non bassi/medi/alti normalizzati: questi ultimi sono
    stirati ciascuno sul proprio range 0..1 e quindi non riflettono la
    vera dominanza relativa, producendo un colore quasi costante invece
    che uno che si sposta chiaramente verso il colore della banda che
    prevale davvero in quell'istante."""
    wb = feat["quota_bassi"][i]
    wm = feat["quota_medi"][i]
    wa = feat["quota_alti"][i]
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

    bassi, medi, alti, quota_bassi, quota_medi, quota_alti = _bande_spettrali(y, sr, hop_length, n_frames)
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
        # quote di dominanza (sommano a 1 ad ogni frame) usate SOLO per il
        # mix colore bassi/medi/alti (_colore_miscelato) — vedi nota in
        # _bande_spettrali sul perche' non si possono riusare bassi/medi/alti
        "quota_bassi": _smussa(quota_bassi),
        "quota_medi": _smussa(quota_medi),
        "quota_alti": _smussa(quota_alti),
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
    # invece di continuare a disegnare a meta' intensita' come prima.
    # Guadagno alzato rispetto alla versione precedente (0.15+0.85*energia,
    # che a energia=1 arrivava solo a ~1.0 pur avendo un tetto a 2.0): con
    # un'escursione piu' ampia i passaggi piano/forte del brano si vedono
    # davvero nell'ampiezza del disegno, non solo in una leggera variazione
    fattore_ampiezza = np.clip((0.10 + 1.6 * energia) * respiro, 0.03, 2.0) * presenza

    k1 = np.clip(2.0 + (4.0 * centroid + 3.5 * alti) * reattivita, 2.0, 9.0)   # timbro/alti
    k2 = np.clip(2.0 + 4.0 * medi * reattivita, 2.0, 9.0)                       # medi

    # per il pattern Fioritura (polare) le due componenti r e a devono condividere
    # la STESSA frequenza secondaria (come nel riferimento BASIC originale,
    # k=3.5 fisso per entrambe): usare due k diversi rompe la correlazione
    # armonica e trasforma i petali puliti in un groviglio denso stile Lissajous
    k_loto = np.clip(2.6 + (1.6 * centroid + 1.1 * alti + 0.8 * medi) * reattivita, 2.2, 5.2)

    # baseline abbassata (era 0.88, escursione solo 0.12) cosi' gli attacchi
    # (onset) producono una variazione di luminosita' chiaramente visibile
    # invece che impercettibile, restando comunque azzerata dal gate nel
    # silenzio
    intensita = np.clip(0.55 + 0.45 * onset, 0.35, 1.0) * presenza
    return fattore_ampiezza, k1, k2, k_loto, onset, intensita, velocita


def disegna_ellisse(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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
                  colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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
                        colore_alti, t1_arr, fps, reattivita=1.0, spessore=1, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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
                        colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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
                   colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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

    # i punti che non scappano mai (interno dell'insieme, valore=0) devono
    # mostrare il colore di SFONDO scelto dall'utente, non nero fisso: prima
    # si moltiplicava solo il colore per il campo (0..1), quindi l'interno
    # risultava sempre nero indipendentemente da colore_bg. Ora si interpola
    # da colore_bg (valore=0) al colore di banda (valore=1), e il gate di
    # silenzio (intensita) fonde ulteriormente verso colore_bg
    colore_arr = np.array(_colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti), dtype=np.float32)
    bg_arr = np.array(colore_bg, dtype=np.float32)
    frac = campo_grande[..., None].astype(np.float32) / 255.0
    mescolato = bg_arr + (colore_arr - bg_arr) * frac
    campo_col = bg_arr + (mescolato - bg_arr) * intensita
    canvas[:] = np.clip(campo_col, 0, 255).astype(np.uint8)

    return canvas


def disegna_aritmia(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="", dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
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


def _punti_campione_lettera_pil(font_pil, carattere, n_punti):
    """Campiona n_punti posizioni (relative all'origine di disegno) sui
    pixel "inchiostrati" di un carattere renderizzato con un font PIL —
    usati per l'effetto particellare: le particelle convergono su questi
    punti per "assemblare" la lettera invece di limitarsi a farla scivolare
    in posizione."""
    try:
        bbox = font_pil.getbbox(carattere)
    except Exception:
        bbox = (0, 0, 20, 20)
    lb, tb, rb, bb = bbox
    larg = max(4, rb - lb + 4)
    alt = max(4, bb - tb + 4)
    img = Image.new("L", (larg, alt), 0)
    ImageDraw.Draw(img).text((-lb + 2, -tb + 2), carattere, font=font_pil, fill=255)
    arr = np.array(img)
    ys, xs = np.where(arr > 80)
    rng_locale = np.random.default_rng(hash(carattere) % (2**31))
    if len(xs) == 0:
        return np.zeros((n_punti, 2)), (-lb + 2, -tb + 2)
    if len(xs) >= n_punti:
        idx = rng_locale.choice(len(xs), size=n_punti, replace=False)
        pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64)
    else:
        idx = rng_locale.integers(0, len(xs), size=n_punti)
        pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64)
        pts += rng_locale.uniform(-1, 1, size=pts.shape)
    pts -= np.array([-lb + 2, -tb + 2])   # relativi all'origine di disegno del testo
    return pts, (-lb + 2, -tb + 2)


def disegna_iscrizione(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                        colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                        dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Una o piu' frasi (separate da a-capo) si materializzano in sequenza
    da uno sciame di particelle che convergono per "assemblare" ogni
    lettera (stile decrittazione/costruzione). Il completamento e' sempre
    calcolato per finire con un margine di sicurezza (3s) prima della fine
    del brano, non esattamente sull'ultimo frame. Con 'sovrapponi' attivo,
    le frasi gia' completate restano a schermo (impilate verticalmente)
    invece di dissolversi, cosi' alla fine si vedono tutte insieme; altrimenti
    ogni frase (tranne l'ultima) si disperde di nuovo in particelle prima
    che inizi la successiva. Font veri (TTF) quando disponibili in una
    cartella "fonts/" accanto allo script, altrimenti ripiega sui font
    Hershey di OpenCV. Con una sola frase, i tempi di comparsa usano i
    battiti reali del brano; con piu' frasi il tempo si divide fra loro in
    proporzione alla lunghezza. Reinterpretazione libera (non copia) della
    texture a caratteri di un pattern C64 BASIC V2 mostrato come riferimento."""
    if not frase:
        return canvas
    elenco_frasi = [f.strip() for f in frase.split("\n") if f.strip()]
    if not elenco_frasi:
        return canvas

    h, w = canvas.shape[:2]
    n_frames_tot = feat["n_frames"]
    margine_finale = int(3 * fps)
    n_frames_utili = max(fps, n_frames_tot - margine_finale)
    colore = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    _fattore, _k1, _k2, _k_loto, _onset, intensita, _velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )

    if stato is None:
        stato = {}

    if "isc_confini" not in stato:
        n_frasi_tot = len(elenco_frasi)
        pesi = np.array([max(1, len(f)) for f in elenco_frasi], dtype=np.float64)
        proporzioni = pesi / pesi.sum()
        durate = np.maximum(1, np.round(proporzioni * n_frames_utili)).astype(np.int64)
        durate[-1] = n_frames_utili - durate[:-1].sum()
        confini = np.concatenate([[0], np.cumsum(durate)])
        confini[-1] = n_frames_tot   # l'ultimo segmento si estende fino alla vera fine
        stato["isc_confini"] = confini
        stato["isc_frasi"] = elenco_frasi
        stato["isc_fase_idx"] = -1
        stato["isc_archivio"] = []       # frasi gia' completate, in modalita' "sovrapponi"
        stato["isc_archiviati"] = set()

        if sovrapponi:
            # dimensione condivisa fra tutte le righe (non una per frase),
            # cosi' le righe impilate hanno lo stesso corpo tipografico
            dim_base = max(10, int((h * 0.85 / max(1, n_frasi_tot)) * dimensione_testo))
            font_prova = _carica_font_ttf(font_scelto, dim_base)
            frase_piu_lunga = max(elenco_frasi, key=len)
            if font_prova is not None:
                bbox = font_prova.getbbox(frase_piu_lunga)
                tw = bbox[2] - bbox[0]
                larghezza_max = w * 0.9 * dimensione_testo
                if tw > larghezza_max and tw > 0:
                    dim_base = max(10, int(dim_base * larghezza_max / tw))
            stato["isc_dim_condivisa"] = dim_base

    confini = stato["isc_confini"]
    frasi = stato["isc_frasi"]
    n_frasi = len(frasi)
    idx_frase = int(np.searchsorted(confini, i, side="right") - 1)
    idx_frase = min(max(idx_frase, 0), n_frasi - 1)
    inizio_seg = int(confini[idx_frase])
    fine_seg_reale = int(confini[idx_frase + 1])
    # la finestra di materializzazione/hold si basa sulla durata "utile"
    # (prima del margine finale), non su fine_seg_reale che puo' essere
    # allungato per coprire la coda di sicurezza
    fine_seg_utile = min(fine_seg_reale, n_frames_utili) if idx_frase == n_frasi - 1 else fine_seg_reale
    durata_seg = max(1, fine_seg_utile - inizio_seg)

    fine_materializza = inizio_seg + int(durata_seg * 0.55)
    if sovrapponi:
        fine_hold = fine_seg_reale   # non dissolve mai in modalita' "sovrapponi"
    else:
        # l'ultima frase non dissolve mai: la sua "hold" si estende fino alla
        # vera fine del segmento (che copre anche il margine di sicurezza),
        # non solo fino alla durata "utile" usata per i calcoli proporzionali
        fine_hold = fine_seg_reale if idx_frase == n_frasi - 1 else inizio_seg + int(durata_seg * 0.85)

    rng_glob = np.random.default_rng(507 + idx_frase)
    n_particelle = 10

    if stato["isc_fase_idx"] != idx_frase:
        frase_corrente = frasi[idx_frase]
        n_car = len(frase_corrente)

        if sovrapponi:
            dimensione_px = stato["isc_dim_condivisa"]
        else:
            dimensione_px = max(10, int(h * 0.20 * dimensione_testo))
        font_pil = _carica_font_ttf(font_scelto, dimensione_px)
        spessore_testo_cv = max(2, spessore + 1)
        stroke_w = max(0, spessore - 1)
        font_cv = _FONT_ISCRIZIONE.get(font_scelto, cv2.FONT_HERSHEY_SIMPLEX)

        # offset verticale di riga, per la modalita' "sovrapponi" (le frasi
        # si dispongono impilate invece che tutte al centro)
        if sovrapponi:
            altezza_riga = dimensione_px * 1.4
            offset_y = (idx_frase - (n_frasi - 1) / 2.0) * altezza_riga
        else:
            offset_y = 0.0
        centro_riga = cy + offset_y

        if font_pil is not None:
            bbox_frase = font_pil.getbbox(frase_corrente)
            tw_f = bbox_frase[2] - bbox_frase[0]
            th_f = bbox_frase[3] - bbox_frase[1]
            if not sovrapponi:
                larghezza_max = w * 0.85 * dimensione_testo
                if tw_f > larghezza_max and tw_f > 0:
                    dimensione_px = max(10, int(dimensione_px * larghezza_max / tw_f))
                    font_pil = _carica_font_ttf(font_scelto, dimensione_px)
                    bbox_frase = font_pil.getbbox(frase_corrente)
                    tw_f = bbox_frase[2] - bbox_frase[0]
                    th_f = bbox_frase[3] - bbox_frase[1]
            x0 = cx - tw_f / 2 - bbox_frase[0]
            y0 = centro_riga - th_f / 2 - bbox_frase[1]
            posizioni_target = []
            for k in range(n_car):
                bbox_pre = font_pil.getbbox(frase_corrente[:k]) if k > 0 else (0, 0, 0, 0)
                posizioni_target.append(np.array([x0 + bbox_pre[2], y0]))
        else:
            (tw_rif, th_rif), _ = cv2.getTextSize(frase_corrente, font_cv, 1.0, spessore_testo_cv)
            altezza_target = h * 0.22 * dimensione_testo if not sovrapponi else dimensione_px
            larghezza_target = w * 0.85 * dimensione_testo
            scala_cv = min(altezza_target / max(th_rif, 1), larghezza_target / max(tw_rif, 1))
            (tw_f, th_f), _ = cv2.getTextSize(frase_corrente, font_cv, scala_cv, spessore_testo_cv)
            x0 = cx - tw_f / 2
            y0 = centro_riga + th_f / 2
            posizioni_target = []
            for k in range(n_car):
                (w_pre, _), _ = cv2.getTextSize(frase_corrente[:k], font_cv, scala_cv, spessore_testo_cv)
                posizioni_target.append(np.array([x0 + w_pre, y0]))
            stato["isc_scala_cv"] = scala_cv

        if n_frasi == 1:
            battiti = feat["battiti_video"]
            battiti = battiti[battiti <= n_frames_utili]
            lock_frames = np.zeros(n_car, dtype=np.int32)
            if len(battiti) >= n_car:
                indici = np.linspace(0, len(battiti) - 1, n_car).astype(int)
                lock_frames[:] = battiti[indici]
            else:
                for k in range(n_car):
                    lock_frames[k] = int(round((k + 1) / n_car * (n_frames_utili - 1)))
        else:
            lock_frames = np.array([
                inizio_seg + int(round((k + 1) / n_car * (fine_materializza - inizio_seg)))
                for k in range(n_car)
            ], dtype=np.int32)

        margine = 40
        pos_correnti = np.array([
            [rng_glob.uniform(margine, max(margine + 1, w - margine)),
             rng_glob.uniform(margine, max(margine + 1, h - margine))]
            for _ in range(n_car)
        ])
        vel = rng_glob.uniform(-1.6, 1.6, size=(n_car, 2))

        n_extra = max(0, int(lettere_extra))
        pos_extra = np.array([[rng_glob.uniform(0, w), rng_glob.uniform(0, h)] for _ in range(n_extra)]) \
            if n_extra > 0 else np.zeros((0, 2))
        vel_extra = rng_glob.uniform(-1.3, 1.3, size=(n_extra, 2)) if n_extra > 0 else np.zeros((0, 2))

        particelle_offset = []
        for ch in frase_corrente:
            if font_pil is not None:
                pts, _ = _punti_campione_lettera_pil(font_pil, ch, n_particelle)
            else:
                pts = np.zeros((n_particelle, 2))
            particelle_offset.append(pts)
        jitter_iniziale = [rng_glob.uniform(-10, 10, size=(n_particelle, 2)) for _ in range(n_car)]

        stato["isc_font_pil"] = font_pil
        stato["isc_font_cv"] = font_cv
        stato["isc_font_dim"] = dimensione_px
        stato["isc_stroke_w"] = stroke_w
        stato["isc_spessore_cv"] = spessore_testo_cv
        stato["isc_frase_corrente"] = frase_corrente
        stato["isc_pos_target"] = posizioni_target
        stato["isc_lock_frames"] = lock_frames
        stato["isc_pos"] = pos_correnti
        stato["isc_pos_congelata"] = [None] * n_car
        stato["isc_vel"] = vel
        stato["isc_bloccato"] = np.zeros(n_car, dtype=bool)
        stato["isc_disperso"] = np.zeros(n_car, dtype=bool)
        stato["isc_vel_dispersione"] = [None] * n_car
        stato["isc_pos_extra"] = pos_extra
        stato["isc_vel_extra"] = vel_extra
        stato["isc_lettera_extra"] = [rng_glob.choice(list(_ALFABETO_ISCRIZIONE)) for _ in range(n_extra)]
        stato["isc_particelle_offset"] = particelle_offset
        stato["isc_jitter_iniziale"] = jitter_iniziale
        stato["isc_rng"] = rng_glob
        stato["isc_fine_materializza"] = fine_materializza
        stato["isc_fine_hold"] = fine_hold
        stato["isc_fine_seg"] = fine_seg_reale

        for k in range(n_car):
            if lock_frames[k] <= i:
                stato["isc_bloccato"][k] = True
                stato["isc_pos"][k] = posizioni_target[k]

        stato["isc_fase_idx"] = idx_frase

    # --- variabili di comodo dallo stato corrente ---
    n_car = len(stato["isc_frase_corrente"])
    rng = stato["isc_rng"]
    transizione = max(1, int(0.5 * fps))
    margine = 20

    def rimbalza(pos, vel):
        for ax, lim in ((0, w), (1, h)):
            if pos[ax] < margine or pos[ax] > lim - margine:
                vel[ax] *= -1

    fine_hold = stato["isc_fine_hold"]
    fine_seg = stato["isc_fine_seg"]

    for k in range(n_car):
        lf = stato["isc_lock_frames"][k]
        if stato["isc_disperso"][k]:
            continue
        if i >= fine_hold and stato["isc_bloccato"][k] and not sovrapponi:
            if stato["isc_vel_dispersione"][k] is None:
                stato["isc_vel_dispersione"][k] = rng.uniform(-3.5, 3.5, size=(len(stato["isc_particelle_offset"][k]), 2))
            continue
        if stato["isc_bloccato"][k]:
            continue
        if i >= lf:
            stato["isc_bloccato"][k] = True
            stato["isc_pos"][k] = stato["isc_pos_target"][k]
        elif i >= lf - transizione:
            if stato["isc_pos_congelata"][k] is None:
                stato["isc_pos_congelata"][k] = stato["isc_pos"][k].copy()
        else:
            rimbalza(stato["isc_pos"][k], stato["isc_vel"][k])
            stato["isc_pos"][k] = stato["isc_pos"][k] + stato["isc_vel"][k]

    # in modalita' "sovrapponi": una volta che tutti i caratteri di questa
    # frase sono bloccati, la archiviamo (disegno statico ad ogni frame
    # successivo, senza piu' bisogno di simularne il vagabondaggio)
    if sovrapponi and idx_frase not in stato["isc_archiviati"] and stato["isc_bloccato"].all():
        stato["isc_archivio"].append({
            "testo": stato["isc_frase_corrente"],
            "pos_target": [p.copy() for p in stato["isc_pos_target"]],
            "font_pil": stato["isc_font_pil"],
            "font_cv": stato["isc_font_cv"],
            "font_dim": stato["isc_font_dim"],
            "stroke_w": stato["isc_stroke_w"],
            "scala_cv": stato.get("isc_scala_cv"),
            "spessore_cv": stato["isc_spessore_cv"],
        })
        stato["isc_archiviati"].add(idx_frase)

    for j in range(len(stato["isc_pos_extra"])):
        rimbalza(stato["isc_pos_extra"][j], stato["isc_vel_extra"][j])
        stato["isc_pos_extra"][j] = stato["isc_pos_extra"][j] + stato["isc_vel_extra"][j]
        if i % 5 == 0:
            stato["isc_lettera_extra"][j] = rng.choice(list(_ALFABETO_ISCRIZIONE))

    # --- disegno: passa a PIL se il font TTF e' disponibile ---
    font_pil = stato["isc_font_pil"]
    usa_pil = font_pil is not None

    if usa_pil:
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
    colore_rgb = (int(colore[2]), int(colore[1]), int(colore[0]))   # BGR -> RGB per PIL
    colore_pieno = tuple(max(0, min(255, int(c * intensita))) for c in colore_rgb)

    # frasi gia' completate e archiviate (modalita' "sovrapponi"): disegno
    # statico, sempre a piena intensita'
    for voce in stato["isc_archivio"]:
        for k, ch in enumerate(voce["testo"]):
            px, py = voce["pos_target"][k]
            if voce["font_pil"] is not None:
                draw.text((float(px), float(py)), ch, font=voce["font_pil"],
                          fill=colore_pieno, stroke_width=voce["stroke_w"], stroke_fill=colore_pieno)
            else:
                cv2.putText(canvas, ch, (int(px), int(py)), voce["font_cv"],
                            voce["scala_cv"], tuple(reversed(colore_pieno)), voce["spessore_cv"], cv2.LINE_AA)

    # lettere decorative di sfondo (sempre disperse/vaganti)
    colore_extra = tuple(max(0, min(255, int(c * intensita * 0.35))) for c in colore_rgb)
    dim_extra = max(8, int(stato["isc_font_dim"] * 0.5))
    font_extra_pil = _carica_font_ttf(font_scelto, dim_extra) if usa_pil else None
    for j in range(len(stato["isc_pos_extra"])):
        x, y = stato["isc_pos_extra"][j]
        if usa_pil and font_extra_pil is not None:
            draw.text((float(x), float(y)), stato["isc_lettera_extra"][j], font=font_extra_pil, fill=colore_extra)
        else:
            cv2.putText(canvas, stato["isc_lettera_extra"][j], (int(x), int(y)), stato["isc_font_cv"],
                        0.5, tuple(reversed(colore_extra)), 1, cv2.LINE_AA)

    for k in range(n_car):
        lf = stato["isc_lock_frames"][k]
        pos_target = stato["isc_pos_target"][k]
        offset_particelle = stato["isc_particelle_offset"][k]

        if stato["isc_disperso"][k] or (i >= fine_hold and stato["isc_bloccato"][k] and not sovrapponi):
            stato["isc_disperso"][k] = True
            t_disp = np.clip((i - fine_hold) / max(1, fine_seg - fine_hold), 0.0, 1.0)
            alpha = max(0.0, 1.0 - t_disp)
            if alpha <= 0.01:
                continue
            vel_disp = stato["isc_vel_dispersione"][k]
            col = tuple(int(c * alpha) for c in colore_pieno)
            for p in range(len(offset_particelle)):
                px = pos_target[0] + offset_particelle[p][0] + vel_disp[p][0] * t_disp * 20
                py = pos_target[1] + offset_particelle[p][1] + vel_disp[p][1] * t_disp * 20
                if usa_pil:
                    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col)
                else:
                    cv2.circle(canvas, (int(px), int(py)), 2, tuple(reversed(col)), -1)
            continue

        if stato["isc_bloccato"][k]:
            if sovrapponi and idx_frase in stato["isc_archiviati"]:
                continue   # gia' disegnata sopra dall'archivio statico
            ch = stato["isc_frase_corrente"][k]
            if usa_pil:
                draw.text((float(pos_target[0]), float(pos_target[1])), ch, font=font_pil,
                          fill=colore_pieno, stroke_width=stato["isc_stroke_w"], stroke_fill=colore_pieno)
            else:
                cv2.putText(canvas, ch, (int(pos_target[0]), int(pos_target[1])), stato["isc_font_cv"],
                            stato["isc_scala_cv"], tuple(reversed(colore_pieno)), stato["isc_spessore_cv"], cv2.LINE_AA)
        elif i >= lf - transizione:
            t_rel = np.clip((i - (lf - transizione)) / transizione, 0.0, 1.0)
            base = stato["isc_pos_congelata"][k] if stato["isc_pos_congelata"][k] is not None else stato["isc_pos"][k]
            jitter = stato["isc_jitter_iniziale"][k]
            for p in range(len(offset_particelle)):
                ox = (1 - t_rel) * jitter[p][0] + t_rel * offset_particelle[p][0]
                oy = (1 - t_rel) * jitter[p][1] + t_rel * offset_particelle[p][1]
                px = (1 - t_rel) * base[0] + t_rel * pos_target[0] + ox
                py = (1 - t_rel) * base[1] + t_rel * pos_target[1] + oy
                if usa_pil:
                    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=colore_pieno)
                else:
                    cv2.circle(canvas, (int(px), int(py)), 2, tuple(reversed(colore_pieno)), -1)
        else:
            x, y = stato["isc_pos"][k]
            jitter = stato["isc_jitter_iniziale"][k]
            for p in range(len(offset_particelle)):
                px, py = x + jitter[p][0], y + jitter[p][1]
                if usa_pil:
                    draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=colore_pieno)
                else:
                    cv2.circle(canvas, (int(px), int(py)), 2, tuple(reversed(colore_pieno)), -1)

    if usa_pil:
        canvas[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return canvas

def disegna_sinapsi(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                     dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Rete di nodi che vagano lentamente per il fotogramma, connessi da
    linee dritte quando sono abbastanza vicini — come una rete neurale o
    un circuito che "si aggrappa" da solo. Struttura interamente diversa
    dalle altre forme: nessuna curva, nessun frattale, solo nodi mobili e
    connessioni rette. Il raggio di connessione pulsa con l'energia del
    brano (fattore incorpora gia' il gate di silenzio: nel silenzio la
    rete quasi si disconnette), la velocita' di deriva dei nodi e' legata
    al BPM. La luminosita' di ogni connessione dipende da quanto i due
    nodi sono vicini (piu' vicini = piu' luminosa)."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    h, w = canvas.shape[:2]
    n_nodi = max(6, len(t1_arr))

    if stato is None:
        stato = {}
    if "sin_pos" not in stato or len(stato["sin_pos"]) != n_nodi:
        rng = np.random.default_rng(507)
        margine = 30
        stato["sin_pos"] = np.array([
            [rng.uniform(margine, max(margine + 1, w - margine)),
             rng.uniform(margine, max(margine + 1, h - margine))]
            for _ in range(n_nodi)
        ])
        stato["sin_vel"] = rng.uniform(-0.7, 0.7, size=(n_nodi, 2))
        stato["sin_rng"] = rng

    pos = stato["sin_pos"]
    vel = stato["sin_vel"]
    rng = stato["sin_rng"]

    vel += rng.uniform(-0.04, 0.04, size=vel.shape)
    vel = np.clip(vel, -1.5, 1.5)
    pos = pos + vel * (0.6 + 0.8 * velocita)

    margine = 20
    for ax, lim in ((0, w), (1, h)):
        fuori = (pos[:, ax] < margine) | (pos[:, ax] > lim - margine)
        vel[fuori, ax] *= -1
        pos[:, ax] = np.clip(pos[:, ax], margine, lim - margine)

    stato["sin_pos"] = pos
    stato["sin_vel"] = vel

    # raggio di connessione pulsante con l'energia (fattore incorpora gia'
    # il gate di silenzio): rete piu' fitta quando il brano e' potente
    raggio_connessione = (min(w, h) * 0.05) + (min(w, h) * 0.18) * np.clip(fattore, 0.0, 1.5)

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    colore_int = tuple(min(int(c * intensita), 255) for c in colore_base)

    for a in range(n_nodi):
        for b in range(a + 1, n_nodi):
            d = dist[a, b]
            if d < raggio_connessione:
                alfa = 1.0 - d / raggio_connessione
                col = tuple(int(c * alfa) for c in colore_int)
                cv2.line(canvas, tuple(pos[a].astype(int)), tuple(pos[b].astype(int)), col, 1, cv2.LINE_AA)

    for p in pos:
        cv2.circle(canvas, tuple(p.astype(int)), max(2, spessore), colore_int, -1, cv2.LINE_AA)

    return canvas


def disegna_labirinto(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                       colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                       dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Griglia di tasselli diagonali (stile Truchet): ogni cella disegna
    una linea "/" o "\\" — ispirato al celeberrimo one-liner Commodore 64
    BASIC "10 PRINT CHR$(205.5+RND(1));:GOTO 10" che genera labirinti
    proprio con questa tecnica (Montfort et al., libro "10 PRINT").
    Reinterpretazione: la probabilita' tra le due diagonali e' pilotata dal
    bilancio bassi/alti del brano invece che puramente casuale (un brano
    piu' basso tende a un orientamento, uno piu' brillante all'altro), e
    la griglia intera si "riorganizza a scatti" ad ogni battito invece di
    restare fissa o cambiare ogni frame."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i]
    alti = feat["alti"][i]
    h, w = canvas.shape[:2]

    n_lato = max(6, int(np.sqrt(len(t1_arr)) * 1.4))
    dim_cella = min(w, h) / n_lato
    n_righe = int(h / dim_cella) + 1
    n_colonne = int(w / dim_cella) + 1

    if stato is None:
        stato = {}
    if "lab_griglia" not in stato or stato["lab_griglia"].shape != (n_righe, n_colonne):
        stato["lab_griglia"] = np.random.random((n_righe, n_colonne)) < 0.5
        stato["lab_prossimo_cambio"] = 0

    if i >= stato["lab_prossimo_cambio"]:
        # probabilita' tra le due diagonali pilotata dal bilancio bassi/alti
        p = np.clip(0.5 + 0.35 * (alti - bassi), 0.1, 0.9)
        stato["lab_griglia"] = np.random.random((n_righe, n_colonne)) < p
        # intervallo di rigenerazione legato al BPM (circa una battuta),
        # con un minimo per non diventare stroboscopico
        intervallo = max(int(fps * 0.25), int(fps * 60.0 / (feat["bpm"] * velocita + 1e-6) * 0.5))
        stato["lab_prossimo_cambio"] = i + intervallo

    griglia = stato["lab_griglia"]
    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    intens_cella = 0.5 + 0.5 * np.clip(fattore, 0.0, 1.5)
    colore_int = tuple(min(int(c * intensita * intens_cella), 255) for c in colore_base)

    for r in range(n_righe):
        y0 = int(r * dim_cella)
        y1 = int((r + 1) * dim_cella)
        for c in range(n_colonne):
            x0 = int(c * dim_cella)
            x1 = int((c + 1) * dim_cella)
            if griglia[r, c]:
                cv2.line(canvas, (x0, y0), (x1, y1), colore_int, spessore, cv2.LINE_AA)
            else:
                cv2.line(canvas, (x1, y0), (x0, y1), colore_int, spessore, cv2.LINE_AA)

    return canvas


def disegna_risonanza(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                       colore_alti, t1_arr, fps, reattivita=1.0, spessore=1, stato=None, frase="",
                       dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Placca di Chladni: punti (come granelli di sabbia) si dispongono
    lungo le linee nodali di un'onda stazionaria — il principio fisico
    della cimatica (Ernst Chladni, XVIII sec.), "il suono reso visibile"
    letteralmente. Formula generica di sovrapposizione di due modi d'onda
    F(x,y)=sin(n*pi*x)*sin(m*pi*y) - rapporto*sin(m*pi*x)*sin(n*pi*y); i
    numeri di modo n,m sono pilotati da bassi/alti del brano, il rapporto
    tra i due termini dai medi — piccole variazioni cambiano drasticamente
    la figura, come un vero piatto di Chladni che cambia frequenza."""
    fattore, _k1, _k2, _k_loto, onset, intensita, _velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]
    h, w = canvas.shape[:2]

    n_modo = 2 + int(6 * bassi)
    m_modo = 2 + int(6 * alti)
    if m_modo == n_modo:
        m_modo += 1
    rapporto = 0.3 + 0.7 * medi

    ris = max(120, int(np.sqrt(len(t1_arr)) * 10))
    xs_lin = np.linspace(0, 1, ris)
    ys_lin = np.linspace(0, 1, ris)
    X, Y = np.meshgrid(xs_lin, ys_lin)

    F = np.sin(n_modo * np.pi * X) * np.sin(m_modo * np.pi * Y) - \
        rapporto * np.sin(m_modo * np.pi * X) * np.sin(n_modo * np.pi * Y)

    soglia = 0.035 + 0.03 * np.clip(fattore, 0.0, 1.5)
    maschera = np.abs(F) < soglia

    ys_idx, xs_idx = np.where(maschera)
    px = (cx + (xs_idx / ris - 0.5) * 2 * raggio_x * 1.4).astype(np.int32)
    py = (cy + (ys_idx / ris - 0.5) * 2 * raggio_y * 1.4).astype(np.int32)

    dentro = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py = px[dentro], py[dentro]

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    intens = 0.7 + 0.3 * onset
    colore_int = np.array([min(int(c * intensita * intens), 255) for c in colore_base], dtype=np.uint8)

    if spessore <= 1:
        canvas[py, px] = colore_int
    else:
        mask_dil = np.zeros((h, w), dtype=np.uint8)
        mask_dil[py, px] = 255
        kernel = np.ones((spessore, spessore), np.uint8)
        mask_dil = cv2.dilate(mask_dil, kernel)
        canvas[mask_dil > 0] = colore_int

    return canvas


def disegna_statica(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                     dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Automa cellulare elementare (Stephen Wolfram, 1983): una riga di
    celle 0/1 genera la riga successiva applicando una regola booleana ai
    tre vicini (sinistra, centro, destra) — da una condizione iniziale
    casuale emerge una texture complessa e imprevedibile, lo stesso
    principio dietro il motivo sul guscio della lumaca Conus textile. La
    regola attiva e' scelta da un piccolo set di regole "caotiche" (non
    frattali auto-simili come la regola 90) in base al timbro del brano;
    l'intera griglia si rigenera a scatti ad ogni battito, come Labirinto,
    non ogni frame (altrimenti risulterebbe stroboscopica)."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i]
    alti = feat["alti"][i]
    h, w = canvas.shape[:2]

    regole_caotiche = [30, 45, 105, 150, 169, 225]

    n_col = max(80, int(np.sqrt(len(t1_arr)) * 8))
    n_righe = min(110, max(60, int(n_col * h / w)))

    if stato is None:
        stato = {}
    if "sta_griglia" not in stato or stato["sta_griglia"].shape != (n_righe, n_col):
        stato["sta_griglia"] = np.zeros((n_righe, n_col), dtype=np.uint8)
        stato["sta_prossimo_cambio"] = 0

    if i >= stato["sta_prossimo_cambio"]:
        idx_regola = int(np.clip((bassi + alti) / 2 * len(regole_caotiche), 0, len(regole_caotiche) - 1))
        regola = regole_caotiche[idx_regola]
        tabella = np.array([(regola >> k) & 1 for k in range(8)], dtype=np.uint8)

        rng = np.random.default_rng(int(i) + 507)
        riga = (rng.random(n_col) < 0.5).astype(np.uint8)
        griglia = np.zeros((n_righe, n_col), dtype=np.uint8)
        for r in range(n_righe):
            griglia[r] = riga
            sx = np.roll(riga, 1)
            dx = np.roll(riga, -1)
            indice = sx * 4 + riga * 2 + dx
            riga = tabella[indice]
        stato["sta_griglia"] = griglia

        intervallo = max(int(fps * 1.0), int(fps * 60.0 / (feat["bpm"] * velocita + 1e-6)))
        stato["sta_prossimo_cambio"] = i + intervallo

    griglia = stato["sta_griglia"]
    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    intens = 0.6 + 0.4 * np.clip(fattore, 0.0, 1.5)
    campo = (griglia * 255).astype(np.uint8)
    campo_grande = cv2.resize(campo, (w, h), interpolation=cv2.INTER_NEAREST)

    colore_finale = tuple(min(255, int(c * intensita * intens)) for c in colore_base)
    canvas[campo_grande > 0] = colore_finale

    return canvas


@functools.lru_cache(maxsize=8)
def _solido_cache(nome):
    """Vertici e spigoli di un solido platonico, calcolati una sola volta
    e messi in cache (non dipendono dall'audio, solo dal nome del
    solido). Gli spigoli sono trovati per distanza minima tra vertici,
    cosi' non serve elencarli a mano per ciascun solido."""
    phi = (1 + np.sqrt(5)) / 2
    if nome == "cubo":
        pts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    elif nome == "ottaedro":
        pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    else:  # icosaedro
        base = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                base.append([0, s1 * 1.0, s2 * phi])
                base.append([s1 * 1.0, s2 * phi, 0])
                base.append([s2 * phi, 0, s1 * 1.0])
        pts = np.unique(np.array(base, dtype=np.float64), axis=0)

    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    triu = np.triu_indices(n, 1)
    d_min = dist[triu].min()
    spigoli = [(int(triu[0][k]), int(triu[1][k])) for k in range(len(triu[0]))
               if dist[triu[0][k], triu[1][k]] < d_min * 1.05]
    return pts, spigoli


def _ruota_3d(pts, ax, ay, az):
    """Ruota un insieme di punti 3D sui tre assi (matrici di rotazione
    classiche, matematica generica non legata ad alcun riferimento)."""
    cx_, sx_ = np.cos(ax), np.sin(ax)
    cy_, sy_ = np.cos(ay), np.sin(ay)
    cz_, sz_ = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
    ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    rz = np.array([[cz_, -sz_, 0], [sz_, cz_, 0], [0, 0, 1]])
    r = rz @ ry @ rx
    return pts @ r.T


def disegna_poliedro(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                      colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                      dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Poliedro 3D in wireframe (cubo/ottaedro/icosaedro) che ruota nello
    spazio, proiettato in 2D con la matematica classica delle matrici di
    rotazione e proiezione prospettica — l'effetto demoscene per
    eccellenza, fatto interamente di spigoli dritti (nessuno degli altri
    motori simula una vera terza dimensione). Il solido attivo cambia a
    scatti ad ogni battito in base a quale banda (bassi/medi/alti) domina
    in quel momento; la velocita' di rotazione sui tre assi e' pilotata
    da bassi/medi/alti separatamente (tumbling asimmetrico), la
    dimensione pulsa con l'energia complessiva, gli spigoli piu' lontani
    (in profondita') sono leggermente piu' tenui per un accenno di
    ombreggiatura pseudo-3D."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]
    h, w = canvas.shape[:2]

    if stato is None:
        stato = {}
    if "pol_angoli" not in stato:
        stato["pol_angoli"] = np.array([0.0, 0.0, 0.0])
        stato["pol_solido"] = "icosaedro"
        stato["pol_prossimo_cambio"] = 0

    if i >= stato["pol_prossimo_cambio"]:
        valori = {"cubo": bassi, "ottaedro": medi, "icosaedro": alti}
        stato["pol_solido"] = max(valori, key=valori.get)
        intervallo = max(int(fps * 1.5), int(fps * 60.0 / (feat["bpm"] * velocita + 1e-6) * 2))
        stato["pol_prossimo_cambio"] = i + intervallo

    stato["pol_angoli"] = stato["pol_angoli"] + np.array([
        0.01 + 0.05 * bassi, 0.01 + 0.05 * medi, 0.01 + 0.05 * alti
    ]) * velocita

    pts, spigoli = _solido_cache(stato["pol_solido"])
    ax_, ay_, az_ = stato["pol_angoli"]
    ruotati = _ruota_3d(pts, ax_, ay_, az_)

    raggio = min(w, h) * 0.28 * (0.7 + 0.5 * np.clip(fattore, 0.0, 1.5))
    distanza_camera = 4.0
    focale = 3.0

    z = ruotati[:, 2]
    fattore_prosp = focale / (z + distanza_camera)
    xs2d = cx + ruotati[:, 0] * raggio * fattore_prosp
    ys2d = cy + ruotati[:, 1] * raggio * fattore_prosp

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)

    for a, b in spigoli:
        z_media = (z[a] + z[b]) / 2
        intens_prof = np.clip(0.5 + 0.5 * (z_media + 2) / 4, 0.3, 1.0)
        colore = tuple(min(int(c * intensita * intens_prof), 255) for c in colore_base)
        p1 = (int(xs2d[a]), int(ys2d[a]))
        p2 = (int(xs2d[b]), int(ys2d[b]))
        cv2.line(canvas, p1, p2, colore, spessore, cv2.LINE_AA)

    return canvas


def disegna_epicicli(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                      colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                      dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Epicicli di Fourier: una catena di cerchi che ruotano l'uno
    sull'altro (il centro di ogni cerchio e' la punta di quello
    precedente) — la rappresentazione geometrica letterale di una serie
    di Fourier, resa popolare da video divulgativi (3blue1brown) ma
    matematica generica, non legata ad alcun riferimento specifico. La
    punta dell'ultimo cerchio lascia una traccia che si accumula nel
    tempo, disegnando un percorso via via piu' intricato — curve
    continue, complementari agli spigoli dritti di Poliedro. Il raggio
    di ogni armonica e' pilotato da una banda diversa del brano (bassi/
    medi/alti/energia complessiva), le velocita' angolari sono multipli
    interi di una velocita' base legata al BPM, come in una vera serie
    di Fourier dove le armoniche piu' alte ruotano piu' veloci."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]
    h, w = canvas.shape[:2]

    if stato is None:
        stato = {}
    if "epi_fase" not in stato:
        stato["epi_fase"] = 0.0
        stato["epi_traccia"] = []

    stato["epi_fase"] += 0.03 * velocita
    fase = stato["epi_fase"]

    n_armoniche = 6
    bande = [bassi, medi, alti, (bassi + medi) / 2, (medi + alti) / 2, np.clip(fattore, 0.0, 1.5) / 1.5]
    raggio_base = min(w, h) * 0.22

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    colore_cerchio = tuple(min(int(c * intensita * 0.35), 255) for c in colore_base)
    colore_traccia = tuple(min(int(c * intensita), 255) for c in colore_base)

    x, y = float(cx), float(cy)
    for k in range(1, n_armoniche + 1):
        r = raggio_base * (0.15 + 0.5 * bande[k - 1]) / k
        ang = fase * k + k * 0.7
        x_next = x + r * np.cos(ang)
        y_next = y + r * np.sin(ang)
        cv2.circle(canvas, (int(x), int(y)), max(1, int(r)), colore_cerchio, 1, cv2.LINE_AA)
        cv2.line(canvas, (int(x), int(y)), (int(x_next), int(y_next)), colore_cerchio, 1, cv2.LINE_AA)
        x, y = x_next, y_next

    stato["epi_traccia"].append((x, y))
    max_traccia = 500
    if len(stato["epi_traccia"]) > max_traccia:
        stato["epi_traccia"] = stato["epi_traccia"][-max_traccia:]

    pts = np.array(stato["epi_traccia"], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) > 1:
        cv2.polylines(canvas, [pts], isClosed=False, color=colore_traccia, thickness=spessore, lineType=cv2.LINE_AA)

    cv2.circle(canvas, (int(x), int(y)), max(2, spessore + 1), colore_traccia, -1, cv2.LINE_AA)

    return canvas


def disegna_plasma(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                    colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                    dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Effetto plasma: il classico effetto demoscene anni '80-'90, campo
    continuo generato sommando onde sinusoidali (piane e radiali) con fasi
    che scorrono nel tempo — matematica generica di dominio pubblico
    (somma di sinusoidi), non legata a un'implementazione specifica. Le
    frequenze delle onde sono pilotate da bassi/medi/alti (piu' turbolento
    quando il brano e' ricco di armoniche), la velocita' di scorrimento
    delle fasi dall'energia complessiva. A differenza di Risonanza
    (soglia binaria sulle linee nodali), qui l'intensita' e' continua,
    dando un flusso morbido che riempie sempre tutto il fotogramma."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i]
    medi = feat["medi"][i]
    alti = feat["alti"][i]
    h, w = canvas.shape[:2]

    ris_w = max(80, int(np.sqrt(len(t1_arr)) * 8))
    ris_h = max(45, int(ris_w * h / w))

    xs = np.linspace(0, 1, ris_w)
    ys = np.linspace(0, 1, ris_h)
    X, Y = np.meshgrid(xs, ys)

    t = t_frame / fps * velocita

    freq1 = 3 + 8 * bassi
    freq2 = 3 + 8 * medi
    freq3 = 4 + 10 * alti
    freq4 = 2 + 4 * np.clip(fattore, 0.0, 1.5)

    dist = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)

    valore = (np.sin(X * freq1 * 2 * np.pi + t * 1.3) +
              np.sin(Y * freq2 * 2 * np.pi + t * 0.9) +
              np.sin(dist * freq3 * 2 * np.pi - t * 1.7) +
              np.sin((X + Y) * freq4 * 2 * np.pi + t * 0.5))
    valore = (valore + 4) / 8

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    campo_u8 = np.clip(valore * 255, 0, 255).astype(np.uint8)
    campo_grande = cv2.resize(campo_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    # come in Julia: le zone del campo a valore basso vanno verso colore_bg
    # (non nero fisso), cosi' lo sfondo scelto dall'utente e' sempre visibile
    # nelle valli dell'interferenza, non solo un colore piatto ignorato
    colore_arr = np.array(colore_base, dtype=np.float32)
    bg_arr = np.array(colore_bg, dtype=np.float32)
    frac = campo_grande[..., None].astype(np.float32) / 255.0
    mescolato = bg_arr + (colore_arr - bg_arr) * frac
    campo_col = bg_arr + (mescolato - bg_arr) * intensita
    canvas[:] = np.clip(campo_col, 0, 255).astype(np.uint8)

    return canvas


def disegna_cometa(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                    colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                    dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Starfield prospettico: ogni stella ha una direzione fissa nel piano
    (scelta una sola volta, in stato) e una profondita' z che diminuisce nel
    tempo, cosi' la stella si avvicina alla camera come nei classici
    starfield demoscene (proiezione prospettica elementare x/z, y/z). Quando
    z scende sotto una soglia la stella "rinasce" lontana con una nuova
    direzione casuale, dando un flusso infinito. Struttura radicalmente
    diversa da Pulviscolo (spirale piatta, senza vera terza dimensione): qui
    e' la profondita' a pilotare sia la posizione a schermo sia la
    dimensione/luminosita' di ogni stella (piu' vicina = piu' grande e
    luminosa, come un vero avvicinamento prospettico)."""
    fattore, _k1, _k2, _k_loto, _onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    alti = feat["alti"][i] * reattivita
    presenza = feat["presenza"][i]
    h, w = canvas.shape[:2]
    n_stelle = max(20, len(t1_arr))

    def _rinasci(n, rng):
        angoli = rng.uniform(0, 2 * np.pi, n)
        raggi = rng.uniform(0.15, 1.0, n)
        direz = np.stack([np.cos(angoli), np.sin(angoli)], axis=1) * raggi[:, None]
        return direz

    if stato is None:
        stato = {}
    if "com_dir" not in stato or len(stato["com_dir"]) != n_stelle:
        rng = np.random.default_rng(507)
        stato["com_dir"] = _rinasci(n_stelle, rng)
        stato["com_z"] = rng.uniform(0.1, 1.0, n_stelle)
        # ordine fisso di attivazione: a bassa energia si accendono solo le
        # prime N di questa permutazione, ad alta energia tutte -- da' un
        # numero di stelle visibili che segue davvero l'energia del brano,
        # senza dover ridimensionare gli array ad ogni frame
        stato["com_ordine"] = rng.permutation(n_stelle)
        stato["com_prev_xy"] = None
        stato["com_rng"] = rng

    dir_ = stato["com_dir"]
    z = stato["com_z"]
    rng = stato["com_rng"]

    # velocita' di avvicinamento ancorata al BPM e rinforzata dall'energia
    # (fattore incorpora gia' il gate di silenzio: nel silenzio le stelle
    # quasi si fermano invece di continuare a sfrecciare)
    dz = (0.006 + 0.03 * np.clip(fattore, 0.0, 1.5)) * velocita
    z = z - dz
    rinate = z <= 0.05
    if rinate.any():
        n_rin = int(rinate.sum())
        dir_[rinate] = _rinasci(n_rin, rng)
        z[rinate] = 1.0
    stato["com_z"] = z
    stato["com_dir"] = dir_

    scala = min(raggio_x, raggio_y) * 1.6
    xs = cx + (dir_[:, 0] / z) * scala
    ys = cy + (dir_[:, 1] / z) * scala

    # frazione di stelle attive: segue energia + gate di presenza, non un
    # numero fisso -- nel silenzio quasi tutte spente, a piena energia tutte
    frazione_attiva = np.clip(0.15 + 0.85 * fattore, 0.05, 1.0) * presenza
    n_attive = max(1, int(n_stelle * frazione_attiva))
    attive = np.zeros(n_stelle, dtype=bool)
    attive[stato["com_ordine"][:n_attive]] = True

    colore_base = _colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti)
    prev_xy = stato["com_prev_xy"]
    if prev_xy is None or len(prev_xy) != n_stelle:
        prev_xy = np.stack([xs, ys], axis=1)

    for k in range(n_stelle):
        if not attive[k]:
            continue
        x, y = xs[k], ys[k]
        if not (0 <= x < w and 0 <= y < h):
            continue
        vicinanza = np.clip(1.0 - z[k], 0.0, 1.0)   # piu' vicina alla camera = piu' grande/luminosa
        spess_stella = max(1, int(spessore * (0.5 + 1.5 * vicinanza)))
        intens_stella = intensita * (0.35 + 0.65 * vicinanza)
        colore = tuple(min(int(c * intens_stella), 255) for c in colore_base)

        px, py = prev_xy[k]
        salto = abs(x - px) + abs(y - py)
        # scia (motion blur) legata agli alti: piu' il brano e' brillante,
        # piu' lunga la striscia -- ma non se la stella e' appena rinata
        # (altrimenti si vede un segmento che attraversa tutto il frame)
        if not rinate[k] and alti > 0.05 and 1 < salto < w * 0.3 and 0 <= px < w and 0 <= py < h:
            cv2.line(canvas, (int(px), int(py)), (int(x), int(y)), colore, spess_stella, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (int(x), int(y)), spess_stella, colore, -1, cv2.LINE_AA)

    stato["com_prev_xy"] = np.stack([xs, ys], axis=1)

    return canvas


def disegna_magma(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                   colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                   dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Metaballs / campi fusi: N sorgenti puntiformi generano ciascuna un
    campo scalare che decade con l'inverso del quadrato della distanza; la
    somma dei campi disegna macchie che si fondono e si separano come gocce
    di lava, invece di semplici cerchi sovrapposti. Le sorgenti sono divise
    in tre gruppi (bassi/medi/alti): il raggio di ciascun gruppo segue
    l'energia della propria banda, e il colore che quel gruppo genera e' il
    colore di banda corrispondente — cosi' la STRUTTURA delle macchie (non
    solo la luminosita') mostra quale frequenza domina in quel momento.
    Reinterpretazione libera: le metaballs sono una tecnica di computer
    graphics classica (Jim Blinn, 1982), non presente nei riferimenti BASIC
    originali. Calcolato a risoluzione ridotta e poi ingrandito con
    interpolazione lineare, per restare performante — stesso principio di
    Frontiera e Plasma."""
    fattore, k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i] * reattivita
    medi = feat["medi"][i] * reattivita
    alti = feat["alti"][i] * reattivita
    _ = fattore, k1  # non usati direttamente qui: energia arriva gia' per banda

    h, w = canvas.shape[:2]
    n_tot = max(6, len(t1_arr))

    if stato is None:
        stato = {}
    if "mag_pos" not in stato or len(stato["mag_pos"]) != n_tot:
        rng = np.random.default_rng(507)
        stato["mag_pos"] = rng.uniform(-0.7, 0.7, (n_tot, 2))
        angoli = rng.uniform(0, 2 * np.pi, n_tot)
        vel_scala = rng.uniform(0.002, 0.006, n_tot)
        stato["mag_vel"] = np.stack([np.cos(angoli), np.sin(angoli)], axis=1) * vel_scala[:, None]
        terzo = n_tot // 3
        gruppo = np.zeros(n_tot, dtype=np.int32)
        gruppo[terzo:2 * terzo] = 1
        gruppo[2 * terzo:] = 2
        rng.shuffle(gruppo)
        stato["mag_gruppo"] = gruppo
        # variazione individuale di raggio: da' organicita', non tutte le
        # macchie di uno stesso gruppo sono identiche
        stato["mag_var"] = rng.uniform(0.75, 1.3, n_tot)

    pos = stato["mag_pos"]
    vel = stato["mag_vel"]
    gruppo = stato["mag_gruppo"]
    var = stato["mag_var"]

    # deriva delle sorgenti: velocita' scalata dal BPM (brani rapidi = macchie
    # piu' inquiete), rimbalzo elastico ai margini dell'area ellittica cosi'
    # le sorgenti non escono mai dal quadro
    pos = pos + vel * velocita
    rimbalzo_x = np.abs(pos[:, 0]) > 1.0
    rimbalzo_y = np.abs(pos[:, 1]) > 1.0
    vel[rimbalzo_x, 0] *= -1
    vel[rimbalzo_y, 1] *= -1
    pos = np.clip(pos, -1.0, 1.0)
    stato["mag_pos"] = pos
    stato["mag_vel"] = vel

    banda_per_gruppo = np.array([bassi, medi, alti], dtype=np.float32)
    raggio_base = min(raggio_x, raggio_y) * 0.24
    raggio_gruppo = raggio_base * (0.45 + 1.35 * np.clip(banda_per_gruppo, 0.0, 1.4))
    raggio_pix = raggio_gruppo[gruppo] * var

    px_i = cx + pos[:, 0] * raggio_x
    py_i = cy + pos[:, 1] * raggio_y

    # risoluzione di calcolo ridotta, legata alle dimensioni del canvas (qui
    # non alla densita': t1_arr controlla il NUMERO di sorgenti, non il
    # dettaglio del campo)
    ris_w = max(90, w // 6)
    ris_h = max(50, h // 6)
    xs_lin = np.linspace(0, w, ris_w, dtype=np.float32)
    ys_lin = np.linspace(0, h, ris_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs_lin, ys_lin)

    dx = grid_x[None, :, :] - px_i[:, None, None].astype(np.float32)
    dy = grid_y[None, :, :] - py_i[:, None, None].astype(np.float32)
    dist2 = dx * dx + dy * dy
    r2 = (raggio_pix.astype(np.float32) ** 2)[:, None, None]
    campo_i = r2 / (dist2 + r2 * 0.25 + 1.0)      # falloff inverso al quadrato, smorzato vicino al centro
    campo = campo_i.sum(axis=0)

    colore_arr = np.zeros((ris_h, ris_w, 3), dtype=np.float32)
    for g, colore_g in enumerate((colore_bassi, colore_medi, colore_alti)):
        maschera = gruppo == g
        if maschera.any():
            colore_arr += campo_i[maschera].sum(axis=0)[..., None] * np.array(colore_g, dtype=np.float32)
    colore_arr /= (campo[..., None] + 1e-6)

    # soglia leggermente "respirante" sugli attacchi: sugli onset le macchie
    # si allargano un istante, invece di restare a dimensione fissa
    soglia = 1.0 - 0.15 * onset
    frac = np.clip((campo - soglia * 0.5) / (soglia * 0.9), 0.0, 1.0)[..., None]

    bg_arr = np.array(colore_bg, dtype=np.float32)
    mescolato = bg_arr + (colore_arr - bg_arr) * frac
    finale = bg_arr + (mescolato - bg_arr) * intensita
    campo_grande = cv2.resize(finale, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[:] = np.clip(campo_grande, 0, 255).astype(np.uint8)

    return canvas


def disegna_mosaico(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                     colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                     dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Mosaico di Voronoi: N semi si muovono lentamente nel piano; ogni pixel
    prende il colore del seme piu' vicino, dividendo il fotogramma in celle
    poligonali che cambiano forma nel tempo — la prima forma "a regioni" del
    catalogo, non piu' punti/linee/curve/campi continui. I semi sono divisi
    in tre gruppi bassi/medi/alti come in Magma, ma qui i confini tra celle
    sono netti (evidenziati con una linea di contrasto, tanto piu' sottile e
    nitida quanto piu' forte e' l'attacco del momento — sugli onset il
    mosaico si "irrigidisce"). Tecnica classica di computer graphics
    (diagramma di Voronoi, Georgy Voronoy 1908), non presente nei
    riferimenti BASIC originali. Calcolato a risoluzione ridotta e
    ingrandito con interpolazione NEAREST (i confini delle celle devono
    restare netti, non sfumare come in Magma/Plasma)."""
    fattore, _k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    _ = fattore

    h, w = canvas.shape[:2]
    n_tot = max(6, len(t1_arr))

    if stato is None:
        stato = {}
    if "mos_pos" not in stato or len(stato["mos_pos"]) != n_tot:
        rng = np.random.default_rng(507)
        stato["mos_pos"] = rng.uniform(-0.85, 0.85, (n_tot, 2))
        angoli = rng.uniform(0, 2 * np.pi, n_tot)
        vel_scala = rng.uniform(0.0015, 0.004, n_tot)
        stato["mos_vel"] = np.stack([np.cos(angoli), np.sin(angoli)], axis=1) * vel_scala[:, None]
        terzo = n_tot // 3
        gruppo = np.zeros(n_tot, dtype=np.int32)
        gruppo[terzo:2 * terzo] = 1
        gruppo[2 * terzo:] = 2
        rng.shuffle(gruppo)
        stato["mos_gruppo"] = gruppo

    pos = stato["mos_pos"]
    vel = stato["mos_vel"]
    gruppo = stato["mos_gruppo"]

    # deriva dei semi: velocita' legata al BPM, rimbalzo elastico ai margini
    pos = pos + vel * velocita
    rimbalzo_x = np.abs(pos[:, 0]) > 1.0
    rimbalzo_y = np.abs(pos[:, 1]) > 1.0
    vel[rimbalzo_x, 0] *= -1
    vel[rimbalzo_y, 1] *= -1
    pos = np.clip(pos, -1.0, 1.0)
    stato["mos_pos"] = pos
    stato["mos_vel"] = vel

    px_i = cx + pos[:, 0] * raggio_x
    py_i = cy + pos[:, 1] * raggio_y

    ris_w = max(100, w // 5)
    ris_h = max(56, h // 5)
    xs_lin = np.linspace(0, w, ris_w, dtype=np.float32)
    ys_lin = np.linspace(0, h, ris_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs_lin, ys_lin)

    dx = grid_x[None, :, :] - px_i[:, None, None].astype(np.float32)
    dy = grid_y[None, :, :] - py_i[:, None, None].astype(np.float32)
    dist2 = dx * dx + dy * dy   # shape (N, ris_h, ris_w)

    ordine = np.argsort(dist2, axis=0)
    idx_vicino = ordine[0]
    idx_secondo = ordine[1]
    dist_vicino = np.take_along_axis(dist2, idx_vicino[None, :, :], axis=0)[0]
    dist_secondo = np.take_along_axis(dist2, idx_secondo[None, :, :], axis=0)[0]
    gruppo_pixel = gruppo[idx_vicino]

    colore_arr = np.zeros((ris_h, ris_w, 3), dtype=np.float32)
    for g, colore_g in enumerate((colore_bassi, colore_medi, colore_alti)):
        colore_arr[gruppo_pixel == g] = np.array(colore_g, dtype=np.float32)

    # confine di cella: dove la distanza dal semee piu' vicino e dal secondo
    # sono quasi uguali. Soglia legata all'onset: sugli attacchi i confini si
    # fanno piu' sottili e netti, nel resto del tempo restano piu' spessi
    scala_confine = min(w, h) / np.sqrt(n_tot)
    margine = np.sqrt(np.maximum(dist_secondo - dist_vicino, 0.0))
    soglia_confine = (0.05 + 0.06 * (1.0 - onset)) * scala_confine
    e_confine = margine < soglia_confine

    # colore del confine per contrasto (chiaro su sfondo scuro, scuro su
    # sfondo chiaro), cosi' resta visibile qualunque sfondo scelga l'utente
    colore_confine = np.array((255, 255, 255) if sum(colore_bg) < 380 else (0, 0, 0), dtype=np.float32)
    colore_arr[e_confine] = colore_arr[e_confine] * 0.25 + colore_confine * 0.75

    bg_arr = np.array(colore_bg, dtype=np.float32)
    finale = bg_arr + (colore_arr - bg_arr) * intensita
    campo_grande = cv2.resize(finale, (w, h), interpolation=cv2.INTER_NEAREST)
    canvas[:] = np.clip(campo_grande, 0, 255).astype(np.uint8)

    return canvas


@functools.lru_cache(maxsize=8)
def _lookup_galleria(ris_w, ris_h, w, h, cx, cy, raggio_x, raggio_y):
    """Lookup (raggio normalizzato, angolo) rispetto al centro, calcolato una
    sola volta e cachato: non dipende dal tempo né dall'audio, solo dalle
    dimensioni/centro del canvas (stesso principio usato per i solidi
    platonici in _solido_cache). Alla base dell'effetto tunnel: ogni frame
    riusa questa stessa griglia, cambia solo come viene "letta" (scroll,
    rotazione, numero di spire)."""
    xs = np.linspace(0, w, ris_w, dtype=np.float64)
    ys = np.linspace(0, h, ris_h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = (grid_x - cx) / raggio_x
    dy = (grid_y - cy) / raggio_y
    raggio_norm = np.sqrt(dx ** 2 + dy ** 2)
    angolo = np.arctan2(dy, dx)
    return raggio_norm, angolo


def disegna_galleria(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                      colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                      dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Tunnel prospettico: lookup (raggio, angolo) rispetto al centro,
    precalcolato una sola volta e cachato (_lookup_galleria, non dipende dal
    tempo né dall'audio — stesso principio usato per i solidi di Poliedro),
    usato per generare anelli concentrici a spirale che alternano colore_bg e
    il colore di banda, scorrendo verso lo spettatore. Effetto BASIC/DOS
    classico (rotozoom/tunnel), non presente nei riferimenti originali. Le
    fasce sono binarie (dentro/fuori), non sfumate: qui non c'e'
    interpolazione morbida come in Magma/Plasma, solo cambi netti, coerente
    con l'estetica "Glitch Brutalista"."""
    fattore, k1, k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i] * reattivita
    medi = feat["medi"][i] * reattivita
    alti = feat["alti"][i] * reattivita
    _ = fattore, k1, k2

    h, w = canvas.shape[:2]
    # risoluzione di calcolo ridotta, scalata dalla densita' (slider) tramite
    # t1_arr — stesso principio di Frontiera: piu' densita' = piu' dettaglio
    # (e piu' costo), non solo un numero di elementi ignorato
    ris_w = max(80, int(np.sqrt(len(t1_arr)) * 6))
    ris_h = max(45, int(ris_w * h / w))

    raggio_norm, angolo = _lookup_galleria(ris_w, ris_h, w, h, cx, cy, raggio_x, raggio_y)

    # numero di anelli -> bassi, numero di spire (twist) -> alti, rotazione
    # continua -> medi (offset) + BPM (velocita' base), scroll verso lo
    # spettatore -> BPM/energia con un balzo in avanti sugli attacchi
    freq_anelli = 3.0 + 7.0 * np.clip(bassi, 0.0, 1.4)
    n_spire = 3.0 + 7.0 * np.clip(alti, 0.0, 1.4)
    rotazione = t_frame * (0.12 + 0.10 * reattivita) * velocita + medi * 1.4
    scroll = t_frame * (0.10 + 0.16 * reattivita) * velocita + onset * 0.7

    pattern = raggio_norm * freq_anelli - scroll + (angolo / (2 * np.pi)) * n_spire + rotazione
    fascia = np.mod(np.floor(pattern), 2.0)

    colore_base = np.array(_colore_miscelato(feat, i, colore_bassi, colore_medi, colore_alti), dtype=np.float32)
    bg_arr = np.array(colore_bg, dtype=np.float32)
    colore_piccolo = np.where(fascia[..., None] > 0.5, colore_base, bg_arr)
    finale = bg_arr + (colore_piccolo - bg_arr) * intensita

    campo_grande = cv2.resize(finale.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    canvas[:] = np.clip(campo_grande, 0, 255).astype(np.uint8)

    return canvas


def disegna_braci(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                   colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                   dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Fuoco algoritmico: il classico fire-effect DOS (righe di "calore" che
    si propagano dal basso verso l'alto, mediando i vicini e raffreddandosi
    leggermente ad ogni passo — tecnica nota almeno dai primi anni '90,
    demoscene e articoli come quelli di Bret Mulvey). Il calore viene
    iniettato alla base in proporzione all'energia del brano, il
    raffreddamento (quanto la fiamma si "spezza"/diventa nervosa risalendo)
    segue gli alti, e la mappatura calore->colore usa i TRE colori di banda
    scelti dall'utente come gradiente (sfondo -> bassi -> medi -> alti)
    invece della classica palette rosso-giallo-bianco. Diverso da tutto il
    resto del catalogo: e' l'unico motore generativo per DIFFUSIONE
    dal basso (Statica e' rule-based su automa, non diffusione continua).
    Calcolato a risoluzione ridotta con stato persistente (il calore deve
    accumularsi frame dopo frame, non puo' essere ricalcolato da zero ad
    ogni frame) e ingrandito con interpolazione lineare."""
    fattore, _k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    energia = np.clip(feat["rms"][i] * 0.6 + feat["bassi"][i] * 0.4, 0.0, 1.4) * reattivita
    alti = feat["alti"][i] * reattivita
    presenza = feat["presenza"][i]
    _ = fattore, onset, velocita

    h, w = canvas.shape[:2]
    ris_w = max(70, int(np.sqrt(len(t1_arr)) * 5))
    ris_h = max(50, int(ris_w * h / w))

    if stato is None:
        stato = {}
    if "bra_calore" not in stato or stato["bra_calore"].shape != (ris_h, ris_w):
        stato["bra_calore"] = np.zeros((ris_h, ris_w), dtype=np.float32)
        stato["bra_rng"] = np.random.default_rng(507)

    calore = stato["bra_calore"]
    rng = stato["bra_rng"]

    # iniezione di calore alla base: poche righe finali, ampiezza legata
    # all'energia complessiva e al gate di presenza (nel silenzio il fuoco
    # si spegne quasi del tutto, invece di continuare a bruciare a meta')
    n_righe_base = max(1, ris_h // 12)
    ampiezza_iniezione = np.clip(0.35 + 0.9 * energia, 0.05, 1.6) * presenza
    calore[-n_righe_base:, :] = rng.uniform(0.0, ampiezza_iniezione, (n_righe_base, ris_w))

    # raffreddamento: piu' alti = fiamma piu' nervosa/frastagliata (si
    # raffredda piu' in fretta risalendo), piu' bassi = fiamma piu' piena
    raffreddamento = np.clip(0.90 - 0.22 * alti, 0.60, 0.96)

    sinistra = np.roll(calore, 1, axis=1)
    destra = np.roll(calore, -1, axis=1)
    sotto1 = np.roll(calore, -1, axis=0)
    sotto2 = np.roll(calore, -2, axis=0)
    nuovo = (sotto1 + sinistra + destra + sotto2) * 0.25 * raffreddamento
    nuovo[-n_righe_base:, :] = calore[-n_righe_base:, :]   # le righe di base restano quelle appena iniettate
    calore = np.clip(nuovo, 0.0, 1.4)
    stato["bra_calore"] = calore

    # mappatura calore -> colore: gradiente a 3 tappe attraverso i colori di
    # banda scelti dall'utente, invece della classica palette rosso-giallo
    t_norm = np.clip(calore / 1.2, 0.0, 1.0)
    bg_arr = np.array(colore_bg, dtype=np.float32)
    cb = np.array(colore_bassi, dtype=np.float32)
    cm = np.array(colore_medi, dtype=np.float32)
    ca = np.array(colore_alti, dtype=np.float32)

    seg = t_norm * 3.0
    f0 = np.clip(seg, 0.0, 1.0)[..., None]
    f1 = np.clip(seg - 1.0, 0.0, 1.0)[..., None]
    f2 = np.clip(seg - 2.0, 0.0, 1.0)[..., None]
    colore_arr = bg_arr + (cb - bg_arr) * f0 + (cm - cb) * f1 + (ca - cm) * f2

    finale = bg_arr + (colore_arr - bg_arr) * intensita
    campo_grande = cv2.resize(finale, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[:] = np.clip(campo_grande, 0, 255).astype(np.uint8)

    return canvas


def disegna_vita(canvas, t_frame, feat, i, cx, cy, raggio_x, raggio_y, colore_bassi, colore_medi,
                  colore_alti, t1_arr, fps, reattivita=1.0, spessore=2, stato=None, frase="",
                  dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False, colore_bg=(0, 0, 0)):
    """Vita: l'automa cellulare di Conway (1970), ma bidimensionale — non una
    riga come Statica, una vera griglia dove ogni cella nasce/sopravvive/
    muore in base a quante vicine ha (8-connesse, topologia toroidale).
    Genera macchie organiche che nascono, oscillano e si spengono, del tutto
    diverso dal profilo a gradini di Statica. La regola attiva alterna fra
    Conway classico (B3/S23) e HighLife (B36/S23, piu' caotica, produce
    "replicatori") in base ai bassi; la velocita' di avanzamento (generazioni
    al secondo) segue il BPM; una piccola probabilita' di mutazione casuale
    per cella, legata agli alti, mantiene la griglia viva invece di farla
    stabilizzare o spegnere troppo in fretta. Il colore di ogni cella dipende
    dalla sua ETA' (generazioni consecutive di vita): nasce nel colore degli
    alti, matura verso i medi, invecchia verso i bassi — un modo diverso di
    leggere l'audio nell'immagine rispetto alla banda-per-gruppo usata in
    Magma/Mosaico."""
    fattore, _k1, _k2, _k_loto, onset, intensita, velocita = _parametri_da_audio(
        feat, i, t_frame, fps, reattivita
    )
    bassi = feat["bassi"][i] * reattivita
    alti = feat["alti"][i] * reattivita
    presenza = feat["presenza"][i]
    _ = fattore, onset

    h, w = canvas.shape[:2]
    ris_w = max(60, int(np.sqrt(len(t1_arr)) * 4))
    ris_h = max(34, int(ris_w * h / w))

    if stato is None:
        stato = {}
    if "vit_griglia" not in stato or stato["vit_griglia"].shape != (ris_h, ris_w):
        rng = np.random.default_rng(507)
        stato["vit_griglia"] = rng.uniform(0, 1, (ris_h, ris_w)) < 0.18
        stato["vit_eta"] = np.zeros((ris_h, ris_w), dtype=np.float32)
        stato["vit_progresso"] = 0.0
        stato["vit_rng"] = rng

    griglia = stato["vit_griglia"]
    eta = stato["vit_eta"]
    rng = stato["vit_rng"]

    # velocita' di avanzamento: generazioni al secondo legate al BPM,
    # accumulate in un contatore frazionario cosi' il ritmo resta fluido
    # anche a fps alti o brani lenti
    gen_per_secondo = (0.8 + 2.2 * reattivita) * velocita
    stato["vit_progresso"] += gen_per_secondo / fps
    passi = 0
    while stato["vit_progresso"] >= 1.0 and passi < 4:
        stato["vit_progresso"] -= 1.0
        passi += 1

    usa_highlife = bassi > 0.55

    for _ in range(passi):
        vicini = np.zeros(griglia.shape, dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                vicini += np.roll(np.roll(griglia, dy, axis=0), dx, axis=1)
        nascita = (vicini == 3) | (usa_highlife & (vicini == 6))
        sopravvive = griglia & ((vicini == 2) | (vicini == 3))
        nuova = nascita | sopravvive

        # mutazione casuale: piccola probabilita' per cella di invertire
        # stato, legata agli alti -- mantiene la griglia viva invece di
        # farla stabilizzare o spegnere del tutto
        p_mutazione = 0.0008 + 0.006 * np.clip(alti, 0.0, 1.4)
        flip = rng.uniform(0, 1, nuova.shape) < p_mutazione
        nuova = nuova ^ flip

        eta_nuova = np.zeros(eta.shape, dtype=np.float32)
        vecchie_sopravvissute = sopravvive & nuova
        eta_nuova[vecchie_sopravvissute] = eta[vecchie_sopravvissute] + 1
        eta = eta_nuova
        griglia = nuova

    # se la popolazione si e' spenta quasi del tutto, la si "ri-accende" con
    # un piccolo seme casuale -- ma solo con presenza audio, cosi' nel
    # silenzio la griglia puo' restare vuota invece di riaccendersi da sola
    popolazione = griglia.mean()
    if popolazione < 0.01 and presenza > 0.15:
        n_semi = max(3, int(griglia.size * 0.02))
        idx_y = rng.integers(0, ris_h, n_semi)
        idx_x = rng.integers(0, ris_w, n_semi)
        griglia[idx_y, idx_x] = True
        eta[idx_y, idx_x] = 0

    stato["vit_griglia"] = griglia
    stato["vit_eta"] = eta

    # colore per cella in base all'eta': nasce negli alti, matura nei medi,
    # invecchia nei bassi
    t_norm = np.clip(eta / 40.0, 0.0, 1.0)
    ca = np.array(colore_alti, dtype=np.float32)
    cm = np.array(colore_medi, dtype=np.float32)
    cb = np.array(colore_bassi, dtype=np.float32)
    seg = t_norm * 2.0
    f0 = np.clip(seg, 0.0, 1.0)[..., None]
    f1 = np.clip(seg - 1.0, 0.0, 1.0)[..., None]
    colore_celle = ca + (cm - ca) * f0 + (cb - cm) * f1

    bg_arr = np.array(colore_bg, dtype=np.float32)
    colore_piccolo = np.where(griglia[..., None], colore_celle, bg_arr)
    finale = bg_arr + (colore_piccolo - bg_arr) * intensita

    campo_grande = cv2.resize(finale.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    canvas[:] = np.clip(campo_grande, 0, 255).astype(np.uint8)

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
    "Sinapsi (rete)": {"funzione": disegna_sinapsi, "n_step": 55, "fade": 0.55},
    "Labirinto (tasselli)": {"funzione": disegna_labirinto, "n_step": 900, "fade": 0.0},
    "Risonanza (placca)": {"funzione": disegna_risonanza, "n_step": 900, "fade": 0.5},
    "Statica (automa)": {"funzione": disegna_statica, "n_step": 900, "fade": 0.0},
    "Poliedro (wireframe)": {"funzione": disegna_poliedro, "n_step": 100, "fade": 0.35},
    "Epicicli (Fourier)": {"funzione": disegna_epicicli, "n_step": 100, "fade": 0.0},
    "Plasma (interferenza)": {"funzione": disegna_plasma, "n_step": 900, "fade": 0.0},
    "Cometa (starfield prospettico)": {"funzione": disegna_cometa, "n_step": 220, "fade": 0.85},
    "Magma (metaballs)": {"funzione": disegna_magma, "n_step": 24, "fade": 0.0},
    "Mosaico (celle di Voronoi)": {"funzione": disegna_mosaico, "n_step": 20, "fade": 0.0},
    "Galleria (tunnel prospettico)": {"funzione": disegna_galleria, "n_step": 900, "fade": 0.0},
    "Braci (fuoco algoritmico)": {"funzione": disegna_braci, "n_step": 900, "fade": 0.0},
    "Vita (automa cellulare 2D)": {"funzione": disegna_vita, "n_step": 900, "fade": 0.0},
}


def genera_video(feat, path_out, width, height, colore_bg, colore_bassi, colore_medi, colore_alti,
                  forma, fps=FPS, seed=507, densita=1.0, spessore=2, reattivita=1.0, frase="",
                  dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False):
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
            dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra, sovrapponi=sovrapponi,
            colore_bg=colore_bg,
        )
        canvas = canvas_u8.astype(np.float32)

        writer.write(canvas_u8)

        if i % 5 == 0 or i == n_frames - 1:
            progress.progress((i + 1) / n_frames, text=f"RENDER :: frame {i+1}/{n_frames}")

    writer.release()
    progress.empty()


def genera_anteprima(feat, width, height, colore_bg, colore_bassi, colore_medi, colore_alti, forma,
                      densita, spessore, reattivita, seed=507, finestra_s=4.0, fps=FPS, frase="",
                      dimensione_testo=1.0, font_scelto=0, lettere_extra=40, sovrapponi=False):
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
            dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra, sovrapponi=sovrapponi,
            colore_bg=colore_bg,
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
                   hex_bg, hex_bassi, hex_medi, hex_alti, densita, spessore, reattivita, seed, vol,
                   frase="", font_scelto=None, sovrapponi=False):
    """Report bilingue IT/EN stile Loop507 (blocco IT completo seguito dal
    blocco EN completo, formato compatto senza separatori — come da
    modello fornito). Restituisce sia la versione da mostrare in chat
    (con code-fence) sia il testo grezzo per il download."""
    riga_extra_it = ""
    riga_extra_en = ""
    if forma == "Iscrizione (testo a tempo)" and frase:
        elenco = [f.strip() for f in frase.split("\n") if f.strip()]
        frasi_join = " / ".join(elenco)
        nome_font = _FONT_TTF_NOMI.get(font_scelto, "n/d")
        riga_extra_it = (
            f"FRASI           :: {frasi_join}\n"
            f"FONT            :: {nome_font}\n"
            f"FRASI IMPILATE  :: {'SI' if sovrapponi else 'NO'}\n"
        )
        riga_extra_en = (
            f"PHRASES         :: {frasi_join}\n"
            f"FONT            :: {nome_font}\n"
            f"STACKED PHRASES :: {'YES' if sovrapponi else 'NO'}\n"
        )

    it = (
        f"[BASICART] // Vol. {vol:03d}\n"
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
        f"{riga_extra_it}"
        "DSP :: RMS, spectral centroid, onset strength, bande bassi/medi/\n"
        "       alti (STFT), beat tracking — nessun modello AI/neurale.\n"
        "Regia e Algoritmo :: Loop507\n"
        "L'audio e' stato scomposto in frequenze. Il codice ne ha ridisegnato la forma."
    )

    en = (
        f"[BASICART] // Vol. {vol:03d}\n"
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
        f"{riga_extra_en}"
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
    sovrapponi = False
    if forma == "Iscrizione (testo a tempo)":
        frase = st.text_area(
            "Frasi da scrivere a tempo (una per riga) :: Phrases to write in time (one per line)",
            max_chars=500, height=120,
            help="Una frase per riga: compaiono in sequenza, il tempo totale si divide fra "
                 "loro in proporzione alla lunghezza :: One phrase per line: they appear in "
                 "sequence, total time is split between them proportionally to length"
        ).upper()

        st.caption("Controlli dedicati :: Dedicated controls — Iscrizione")
        colA, colB, colC = st.columns(3)
        with colA:
            dimensione_testo = st.slider(
                "Dimensione testo :: Text size", 0.5, 3.0, 1.0, 0.1
            )
        with colB:
            nomi_font = {
                0: "Geist Mono (tecnico)", 1: "Big Shoulders (bold)",
                2: "IBM Plex Serif (elegante)", 3: "Boldonse (pesante)",
                4: "Erica One (rotondo)",
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
        sovrapponi = st.checkbox(
            "Le frasi restano impilate a schermo (non si dissolvono) :: "
            "Phrases stay stacked on screen (don't dissolve)",
            value=False,
            help="Con piu' righe, ogni frase completata resta visibile sopra/sotto le altre "
                 "invece di sparire prima della successiva :: With multiple lines, each "
                 "completed phrase stays visible above/below the others instead of "
                 "disappearing before the next one"
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
                dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra, sovrapponi=sovrapponi,
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
                    dimensione_testo=dimensione_testo, font_scelto=font_scelto, lettere_extra=lettere_extra, sovrapponi=sovrapponi,
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
                    seed=507, vol=vol, frase=frase, font_scelto=font_scelto, sovrapponi=sovrapponi,
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
