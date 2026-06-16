"""
ssb_modulator.py
Módulo de modulación/demodulación SSB e ISB usando la Transformada de Hilbert.
Incluye TX por parlante y RX por micrófono.
EL5522 - Taller de Comunicaciones Eléctricas
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import scipy.io.wavfile as wav
import sounddevice as sd


# ─────────────────────────────────────────────
# Utilidades de audio
# ─────────────────────────────────────────────

def load_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Carga un archivo WAV y normaliza a float64 en [-1, 1]."""
    fs, data = wav.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]          # Tomar canal izquierdo si es estéreo
    data = data.astype(np.float64)
    data /= np.max(np.abs(data)) + 1e-12
    return data, fs


def save_wav(filepath: str, signal: np.ndarray, fs: int):
    """Guarda una señal como archivo WAV (16-bit)."""
    signal = signal / (np.max(np.abs(signal)) + 1e-12)
    signal_int = (signal * 32767).astype(np.int16)
    wav.write(filepath, fs, signal_int)


def lowpass_filter(signal: np.ndarray, cutoff: float, fs: int, order: int = 6) -> np.ndarray:
    """Filtro pasa-bajos Butterworth."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────
# Modulación SSB
# ─────────────────────────────────────────────

def modulate_ssb(message: np.ndarray, fs: int, fc: float,
                 mode: str = "SSB-SC", sideband: str = "USB",
                 Ac: float = 1.0) -> dict:
    """
    Modula una señal de mensaje usando SSB (método de la fase / Hilbert).

    Parámetros
    ----------
    message  : señal de mensaje normalizada
    fs       : frecuencia de muestreo [Hz]
    fc       : frecuencia de portadora [Hz]
    mode     : "SSB-SC" | "SSB-FC" | "ISB"
    sideband : "USB" | "LSB"  (ignorado para ISB)
    Ac       : amplitud de portadora

    Retorna dict con claves: message, dsb, ssb (o isb), t, fs, fc, mode, sideband
    """
    t = np.arange(len(message)) / fs
    carrier = np.cos(2 * np.pi * fc * t)

    # Transformada de Hilbert del mensaje
    analytic  = hilbert(message)
    msg_hat   = np.imag(analytic)    # señal cuadratura (90°)

    # DSB-SC
    dsb_sc = message * carrier

    # SSB-SC
    if sideband == "USB":
        ssb_sc = 0.5 * (message * np.cos(2 * np.pi * fc * t)
                        - msg_hat * np.sin(2 * np.pi * fc * t))
    else:  # LSB
        ssb_sc = 0.5 * (message * np.cos(2 * np.pi * fc * t)
                        + msg_hat * np.sin(2 * np.pi * fc * t))

    # Portadora completa (SSB-FC agrega portadora)
    if mode == "SSB-FC":
        ssb = ssb_sc + Ac * carrier
        dsb = dsb_sc + Ac * carrier
    else:
        ssb = ssb_sc
        dsb = dsb_sc

    return {
        "message": message,
        "dsb": dsb,
        "ssb": ssb,
        "t": t,
        "fs": fs,
        "fc": fc,
        "mode": mode,
        "sideband": sideband,
    }


def modulate_isb(msg_upper: np.ndarray, msg_lower: np.ndarray,
                 fs: int, fc: float, Ac: float = 1.0) -> dict:
    """
    Modulación ISB (Independent Sideband):
    USB ← msg_upper  |  LSB ← msg_lower
    """
    n = min(len(msg_upper), len(msg_lower))
    msg_upper, msg_lower = msg_upper[:n], msg_lower[:n]

    t = np.arange(n) / fs
    usb = modulate_ssb(msg_upper, fs, fc, mode="SSB-SC", sideband="USB")["ssb"]
    lsb = modulate_ssb(msg_lower, fs, fc, mode="SSB-SC", sideband="LSB")["ssb"]
    isb = usb + lsb + Ac * np.cos(2 * np.pi * fc * t)

    return {
        "msg_upper": msg_upper,
        "msg_lower": msg_lower,
        "usb": usb,
        "lsb": lsb,
        "isb": isb,
        "t": t,
        "fs": fs,
        "fc": fc,
    }


# ─────────────────────────────────────────────
# Demodulación SSB
# ─────────────────────────────────────────────

def demodulate_ssb(ssb: np.ndarray, fs: int, fc: float,
                   phase_error_deg: float = 0.0,
                   freq_error_hz: float = 0.0,
                   sideband: str = "USB") -> dict:
    """
    Demodula una señal SSB con posibles errores de fase y frecuencia.

    Parámetros
    ----------
    ssb             : señal SSB recibida
    fs              : frecuencia de muestreo
    fc              : frecuencia de portadora nominal
    phase_error_deg : error de fase en grados
    freq_error_hz   : error de frecuencia en Hz
    sideband        : "USB" | "LSB"
    """
    t = np.arange(len(ssb)) / fs
    phi  = np.deg2rad(phase_error_deg)
    fc_r = fc + freq_error_hz           # portadora local con error

    # Portadora local con errores
    local_carrier = np.cos(2 * np.pi * fc_r * t + phi)

    # Mezcla con portadora local
    mixed = ssb * local_carrier

    # Filtro pasa-bajos para recuperar banda base
    bw = fc * 0.6          # ancho de banda conservador
    recovered = lowpass_filter(mixed, bw, fs)
    recovered *= 2.0       # compensar pérdida de potencia

    # Normalizar
    peak = np.max(np.abs(recovered))
    if peak > 1e-10:
        recovered /= peak

    return {
        "recovered": recovered,
        "t": t,
        "fs": fs,
        "fc": fc,
        "phase_error_deg": phase_error_deg,
        "freq_error_hz": freq_error_hz,
        "sideband": sideband,
    }


def demodulate_isb(isb: np.ndarray, fs: int, fc: float,
                   phase_error_deg: float = 0.0,
                   freq_error_hz: float = 0.0) -> dict:
    """Demodula ambas bandas de una señal ISB."""
    usb_result = demodulate_ssb(isb, fs, fc, phase_error_deg, freq_error_hz, "USB")
    lsb_result = demodulate_ssb(isb, fs, fc, phase_error_deg, freq_error_hz, "LSB")

    return {
        "recovered_upper": usb_result["recovered"],
        "recovered_lower": lsb_result["recovered"],
        "t": usb_result["t"],
        "fs": fs,
    }


# ─────────────────────────────────────────────
# TX: reproducir señal SSB por parlante
# ─────────────────────────────────────────────

def transmit_ssb(mod_result: dict, progress_cb=None) -> float:
    """
    Reproduce la señal SSB modulada por el parlante de la PC.

    La señal ya está en banda de audio (fc < 20 kHz), así que se
    reproduce directamente sin ninguna conversión extra.

    Retorna la duración en segundos.
    """
    ssb = mod_result["ssb"]
    fs  = mod_result["fs"]
    fc  = mod_result["fc"]

    # Normalizar para evitar clipping
    ssb_norm = ssb / (np.max(np.abs(ssb)) + 1e-12)

    duration = len(ssb_norm) / fs

    if progress_cb:
        progress_cb(f"Transmitiendo SSB por parlante — fc={fc} Hz, {duration:.1f}s...")

    sd.play(ssb_norm.astype(np.float32), samplerate=fs)
    sd.wait()

    if progress_cb:
        progress_cb("✓ Transmisión SSB terminada.")

    return duration


# ─────────────────────────────────────────────
# RX: grabar por micrófono y demodular
# ─────────────────────────────────────────────

def receive_ssb(fs: int, fc: float, duration_s: float,
                sideband: str = "USB",
                phase_error_deg: float = 0.0,
                freq_error_hz: float = 0.0,
                progress_cb=None) -> dict:
    """
    Graba audio por el micrófono durante duration_s segundos,
    luego demodula la señal SSB recibida.

    La señal captada ya viene en banda de audio, así que se
    demodula directamente (igual que si fuera simulada).

    Retorna dict con: ssb_rx, recovered, t, fs, fc, ...
    """
    if progress_cb:
        progress_cb(f"Grabando {duration_s}s por micrófono (fs={fs} Hz)...")

    n_samples = int(duration_s * fs)
    recorded  = sd.rec(n_samples, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    ssb_rx = recorded.flatten().astype(np.float64)

    # Normalizar lo grabado
    peak = np.max(np.abs(ssb_rx))
    if peak > 1e-6:
        ssb_rx /= peak

    if progress_cb:
        progress_cb("Demodulando señal recibida...")

    demod = demodulate_ssb(ssb_rx, fs, fc,
                           phase_error_deg=phase_error_deg,
                           freq_error_hz=freq_error_hz,
                           sideband=sideband)
    demod["ssb_rx"] = ssb_rx
    return demod
