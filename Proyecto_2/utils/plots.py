"""
plots.py
Funciones de graficación para SSB y modulación digital.
EL5522 - Taller de Comunicaciones Eléctricas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from scipy.signal import welch


# ─────────────────────────────────────────────
# Utilidades FFT
# ─────────────────────────────────────────────

def compute_spectrum(signal: np.ndarray, fs: int,
                     n_fft: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Retorna frecuencias y magnitud espectral (dB) bilateral."""
    n = len(signal)
    if n_fft is None:
        n_fft = n
    window = np.hanning(n)
    X = np.fft.fftshift(np.fft.fft(signal * window, n=n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1/fs))
    mag_db = 20 * np.log10(np.abs(X) / n + 1e-12)
    return freqs, mag_db


# ─────────────────────────────────────────────
# Gráficas SSB
# ─────────────────────────────────────────────

def plot_ssb_modulator(result: dict) -> Figure:
    """
    4 subgráficas: tiempo y frecuencia del mensaje,
    espectro DSB, espectro SSB/ISB.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    colors = {
        'message': '#4fc3f7',
        'dsb':     '#ff8a65',
        'ssb':     '#a5d6a7',
        'carrier': '#ce93d8',
    }

    t       = result["t"]
    message = result["message"]
    dsb     = result["dsb"]
    ssb     = result["ssb"]
    fs      = result["fs"]
    fc      = result["fc"]
    mode    = result["mode"]
    sb      = result.get("sideband", "")

    # Limitar a 50 ms para visualización
    t_max   = min(0.05, t[-1])
    mask    = t <= t_max

    def ax_style(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=9, pad=6)
        ax.tick_params(colors='#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    # ── Tiempo: Mensaje ──────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t[mask]*1e3, message[mask], color=colors['message'], lw=0.9)
    ax0.set_xlabel("Tiempo [ms]")
    ax0.set_ylabel("Amplitud")
    ax_style(ax0, "Mensaje m(t)")

    # ── Espectro: Mensaje ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    f_msg, M = compute_spectrum(message, fs)
    mask_f = np.abs(f_msg) <= fc * 1.5
    ax1.plot(f_msg[mask_f]/1e3, M[mask_f], color=colors['message'], lw=0.9)
    ax1.set_xlabel("Frecuencia [kHz]")
    ax1.set_ylabel("Magnitud [dB]")
    ax_style(ax1, "Espectro del Mensaje M(f)")

    # ── Espectro: DSB ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    f_dsb, D = compute_spectrum(dsb, fs)
    mask_d = (np.abs(f_dsb) >= fc * 0.3) & (np.abs(f_dsb) <= fc * 1.8)
    ax2.plot(f_dsb[mask_d]/1e3, D[mask_d], color=colors['dsb'], lw=0.9)
    ax2.axvline(fc/1e3,  color='white', lw=0.5, linestyle='--', alpha=0.4)
    ax2.axvline(-fc/1e3, color='white', lw=0.5, linestyle='--', alpha=0.4)
    ax2.set_xlabel("Frecuencia [kHz]")
    ax2.set_ylabel("Magnitud [dB]")
    ax_style(ax2, f"Espectro DSB ({mode})")

    # ── Espectro: SSB ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    f_ssb, S = compute_spectrum(ssb, fs)
    ax3.plot(f_ssb[mask_d]/1e3, S[mask_d], color=colors['ssb'], lw=0.9)
    ax3.axvline(fc/1e3,  color='white', lw=0.5, linestyle='--', alpha=0.4)
    ax3.axvline(-fc/1e3, color='white', lw=0.5, linestyle='--', alpha=0.4)
    ax3.set_xlabel("Frecuencia [kHz]")
    ax3.set_ylabel("Magnitud [dB]")
    label = f"ISB" if mode == "ISB" else f"SSB ({sb}, {mode})"
    ax_style(ax3, f"Espectro {label} — fc={fc/1e3:.1f} kHz")

    # ── Tiempo: SSB ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(t[mask]*1e3, ssb[mask], color=colors['ssb'], lw=0.8,
             label=f"SSB ({sb})")
    ax4.plot(t[mask]*1e3, dsb[mask], color=colors['dsb'], lw=0.6,
             alpha=0.5, label="DSB")
    ax4.set_xlabel("Tiempo [ms]")
    ax4.set_ylabel("Amplitud")
    ax4.legend(facecolor='#161b22', labelcolor='white', fontsize=8)
    ax_style(ax4, "Señales SSB y DSB en el tiempo")

    fig.suptitle(f"Modulación SSB — Método Hilbert | fc = {fc/1e3:.2f} kHz | {mode} {sb}",
                 color='white', fontsize=12, y=0.98)
    return fig


def plot_ssb_demodulator(mod_result: dict, demod_result: dict) -> Figure:
    """Gráficas del demodulador SSB: tiempo y frecuencia del mensaje recuperado."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f"Demodulación SSB | Δφ={demod_result['phase_error_deg']:.1f}° | "
        f"Δf={demod_result['freq_error_hz']:.1f} Hz",
        color='white', fontsize=11, y=0.98
    )

    original  = mod_result["message"]
    recovered = demod_result["recovered"]
    t         = demod_result["t"]
    fs        = demod_result["fs"]
    n = min(len(original), len(recovered))

    t_show = min(0.05, t[-1])
    mask   = t[:n] <= t_show

    def ax_s(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=9)
        ax.tick_params(colors='#8b949e')
        for sp in ['bottom','left']: ax.spines[sp].set_color('#30363d')
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    axes[0,0].plot(t[:n][mask]*1e3, original[:n][mask],  color='#4fc3f7', lw=0.9)
    axes[0,0].set_xlabel("Tiempo [ms]"); axes[0,0].set_ylabel("Amplitud")
    ax_s(axes[0,0], "Mensaje original")

    axes[0,1].plot(t[:n][mask]*1e3, recovered[:n][mask], color='#a5d6a7', lw=0.9)
    axes[0,1].set_xlabel("Tiempo [ms]"); axes[0,1].set_ylabel("Amplitud")
    ax_s(axes[0,1], "Mensaje recuperado")

    f_orig, Mo = compute_spectrum(original[:n],  fs)
    f_rec,  Mr = compute_spectrum(recovered[:n], fs)
    fc = mod_result["fc"]
    flim = fc * 0.7

    mask_f = np.abs(f_orig) <= flim
    axes[1,0].plot(f_orig[mask_f]/1e3, Mo[mask_f], color='#4fc3f7', lw=0.9)
    axes[1,0].set_xlabel("Frecuencia [kHz]"); axes[1,0].set_ylabel("Magnitud [dB]")
    ax_s(axes[1,0], "Espectro original")

    mask_r = np.abs(f_rec) <= flim
    axes[1,1].plot(f_rec[mask_r]/1e3, Mr[mask_r], color='#a5d6a7', lw=0.9)
    axes[1,1].set_xlabel("Frecuencia [kHz]"); axes[1,1].set_ylabel("Magnitud [dB]")
    ax_s(axes[1,1], "Espectro recuperado")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Gráficas modulación digital
# ─────────────────────────────────────────────

def plot_digital_tx(audio: np.ndarray, fs: int,
                    f_mark: float, f_space: float) -> Figure:
    """Tiempo y espectro de la señal FSK transmitida. Constelación BFSK."""
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    def ax_s(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=9)
        ax.tick_params(colors='#8b949e')
        for sp in ['bottom','left']: ax.spines[sp].set_color('#30363d')
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    # Tiempo (primeros 50ms)
    t    = np.arange(len(audio)) / fs
    t_ms = t * 1e3
    n_show = min(int(0.05 * fs), len(audio))
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(t_ms[:n_show], audio[:n_show], color='#4fc3f7', lw=0.7)
    ax0.set_xlabel("Tiempo [ms]"); ax0.set_ylabel("Amplitud")
    ax_s(ax0, "Señal BFSK transmitida (primeros 50ms)")

    # Espectro
    ax1 = fig.add_subplot(gs[1, :2])
    freqs, mag = compute_spectrum(audio[:min(len(audio), 44100)], fs)
    mask = (freqs >= 0) & (freqs <= 5000)
    ax1.plot(freqs[mask], mag[mask], color='#ff8a65', lw=0.9)
    ax1.axvline(f_mark,  color='#a5d6a7', lw=1.5, linestyle='--', label=f'Mark {f_mark} Hz')
    ax1.axvline(f_space, color='#ce93d8', lw=1.5, linestyle='--', label=f'Space {f_space} Hz')
    ax1.legend(facecolor='#161b22', labelcolor='white', fontsize=8)
    ax1.set_xlabel("Frecuencia [Hz]"); ax1.set_ylabel("Magnitud [dB]")
    ax_s(ax1, "Espectro BFSK")

    # Constelación BFSK (I/Q simplificada)
    ax2 = fig.add_subplot(gs[:, 2])
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), '--', color='#30363d', lw=0.6)
    ax2.scatter([0, 0], [1, -1], s=180, zorder=5,
                c=['#a5d6a7', '#ce93d8'], edgecolors='white', linewidths=0.8)
    ax2.annotate(f'Mark\n{f_mark} Hz', (0, 1),   color='#a5d6a7', ha='center', va='bottom', fontsize=8)
    ax2.annotate(f'Space\n{f_space} Hz', (0, -1), color='#ce93d8', ha='center', va='top',    fontsize=8)
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel("I"); ax2.set_ylabel("Q")
    ax_s(ax2, "Constelación BFSK")

    fig.suptitle(f"Modulador BFSK | Mark={f_mark}Hz | Space={f_space}Hz | {len(audio)//fs}s de audio",
                 color='white', fontsize=11)
    return fig


def plot_ber_and_eye(ber_data: dict, audio: np.ndarray,
                     fs: int, spb: int) -> Figure:
    """Curva BER vs SNR y diagrama de ojo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('#0d1117')

    def ax_s(ax, title):
        ax.set_facecolor('#161b22')
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='#8b949e')
        for sp in ['bottom','left']: ax.spines[sp].set_color('#30363d')
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    # BER
    snr = ber_data["snr_db"]
    ax1.semilogy(snr, ber_data["ber_theoretical"], color='#ff8a65',
                 lw=2, linestyle='--', label='Teórico')
    ax1.semilogy(snr, ber_data["ber_simulated"],   color='#4fc3f7',
                 lw=2, label='Simulado')
    ax1.set_xlabel("Eb/N0 [dB]"); ax1.set_ylabel("BER")
    ax1.legend(facecolor='#161b22', labelcolor='white')
    ax1.grid(True, which='both', color='#21262d', linestyle='--', lw=0.5)
    ax_s(ax1, "Curva BER — BFSK no-coherente")

    # Diagrama de ojo (superponer períodos de bit)
    n_bits_eye = min(200, len(audio) // spb)
    eye_data = audio[:n_bits_eye * spb].reshape(n_bits_eye, spb)
    t_eye = np.linspace(0, 2, spb)
    for row in eye_data:
        ax2.plot(t_eye, row, color='#a5d6a7', lw=0.4, alpha=0.4)
    ax2.set_xlabel("Tiempo normalizado [T_b]")
    ax2.set_ylabel("Amplitud")
    ax2.axvline(1.0, color='#ff8a65', lw=1, linestyle='--', alpha=0.7, label='Centro de ojo')
    ax2.legend(facecolor='#161b22', labelcolor='white', fontsize=8)
    ax_s(ax2, "Diagrama de Ojo — BFSK")

    fig.suptitle("Análisis de rendimiento — Modulación Digital BFSK",
                 color='white', fontsize=12)
    plt.tight_layout()
    return fig
