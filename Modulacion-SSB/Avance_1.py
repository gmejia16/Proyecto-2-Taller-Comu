"""
Carlos Gutierrez
Adrian Parajeles
Gustavo Padilla
Geovanny Mejía


Taller de Comunicaciones Eléctricas
Avance 1: Carga de archivo WAV + Modulador SSB inicial

Ejecutar:
    python -m venv venv
    source venv/bin/activate
    pip install numpy scipy matplotlib
    python avance1.py
"""

import numpy as np
from scipy.signal import hilbert
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ── 1. Cargar archivo WAV ─────────────────────────────────────

def cargar_wav(filepath):
    fs, data = wav.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]                      # estéreo → mono
    data = data.astype(np.float64)
    data /= np.max(np.abs(data)) + 1e-12       # normalizar [-1, 1]
    return data, fs

# ── 2. Modulador SSB (Transformada de Hilbert) ────────────────

def modular_ssb(mensaje, fs, fc, banda="USB"):
    t       = np.arange(len(mensaje)) / fs
    msg_hat = np.imag(hilbert(mensaje))        # señal en cuadratura

    # DSB-SC: m(t)·cos(2π·fc·t)
    dsb = mensaje * np.cos(2 * np.pi * fc * t)

    # USB: m(t)cos(wct) − m̂(t)sin(wct)
    # LSB: m(t)cos(wct) + m̂(t)sin(wct)
    if banda == "USB":
        ssb = 0.5 * (mensaje * np.cos(2 * np.pi * fc * t)
                     - msg_hat * np.sin(2 * np.pi * fc * t))
    else:
        ssb = 0.5 * (mensaje * np.cos(2 * np.pi * fc * t)
                     + msg_hat * np.sin(2 * np.pi * fc * t))

    return t, dsb, ssb

# ── 3. Gráficas ───────────────────────────────────────────────

def graficar(mensaje, t, dsb, ssb, fs, fc, banda):
    def espectro(s):
        n   = len(s)
        X   = np.fft.fftshift(np.fft.fft(s * np.hanning(n)))
        f   = np.fft.fftshift(np.fft.fftfreq(n, 1/fs))
        dB  = 20 * np.log10(np.abs(X) / n + 1e-12)
        return f, dB

    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"Modulador SSB — fc={fc} Hz — {banda}", fontsize=12)
    t_ms  = t * 1e3
    n50ms = int(0.05 * fs)           # primeros 50 ms

    # Tiempo
    axs[0,0].plot(t_ms[:n50ms], mensaje[:n50ms])
    axs[0,0].set(title="Mensaje m(t)", xlabel="ms", ylabel="Amplitud")

    axs[0,1].plot(t_ms[:n50ms], dsb[:n50ms])
    axs[0,1].set(title="DSB", xlabel="ms", ylabel="Amplitud")

    axs[0,2].plot(t_ms[:n50ms], ssb[:n50ms])
    axs[0,2].set(title=f"SSB ({banda})", xlabel="ms", ylabel="Amplitud")

    # Espectro
    f_m, M = espectro(mensaje)
    f_d, D = espectro(dsb)
    f_s, S = espectro(ssb)
    rango  = np.abs(f_m) <= fc * 2

    axs[1,0].plot(f_m[rango]/1e3, M[rango])
    axs[1,0].set(title="Espectro M(f)", xlabel="kHz", ylabel="dB")

    axs[1,1].plot(f_d[rango]/1e3, D[rango])
    axs[1,1].set(title="Espectro DSB", xlabel="kHz", ylabel="dB")

    axs[1,2].plot(f_s[rango]/1e3, S[rango])
    axs[1,2].set(title=f"Espectro SSB ({banda})", xlabel="kHz", ylabel="dB")

    plt.tight_layout()
    plt.show()

# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Seleccionar archivo WAV
    Tk().withdraw()
    ruta = filedialog.askopenfilename(
        title="Seleccionar archivo WAV",
        filetypes=[("WAV", "*.wav")]
    )
    if not ruta:
        print("No se seleccionó ningún archivo.")
        exit()

    # Parámetros
    fc    = float(input("Frecuencia portadora fc [Hz]: "))
    banda = input("Banda lateral (USB / LSB): ").strip().upper()
    if banda not in ("USB", "LSB"):
        banda = "USB"

    # Procesar
    mensaje, fs = cargar_wav(ruta)
    print(f"WAV cargado: fs={fs} Hz, duración={len(mensaje)/fs:.2f}s")

    t, dsb, ssb = modular_ssb(mensaje, fs, fc, banda)
    print("Modulación SSB completada.")

    graficar(mensaje, t, dsb, ssb, fs, fc, banda)
