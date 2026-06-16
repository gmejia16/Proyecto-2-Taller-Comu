"""
app.py — Aplicación principal
Interfaz gráfica con Tkinter para:
  - Modulación/Demodulación SSB (Hilbert)
  - Modulación Digital Pasobanda (BFSK audio)

EL5522 - Taller de Comunicaciones Eléctricas
Ejecutar: python app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import sys
import os

# Asegurar que los módulos del proyecto estén en el path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from core.ssb_modulator  import (load_wav, save_wav, modulate_ssb,
                                  modulate_isb, demodulate_ssb, demodulate_isb,
                                  transmit_ssb, receive_ssb)
from core.digital_modem  import (build_frame, audio_to_bits, decode_frame,
                                  compute_ber_curve, transmit, receive,
                                  save_modulated_wav, receive_from_file,
                                  add_awgn,
                                  FS, F_MARK, F_SPACE, SPB)
from utils.plots          import (plot_ssb_modulator, plot_ssb_demodulator,
                                  plot_digital_tx, plot_ber_and_eye)


# ═══════════════════════════════════════════════════════════════
# Paleta y estilos
# ═══════════════════════════════════════════════════════════════

BG      = "#0d1117"
BG2     = "#161b22"
BG3     = "#21262d"
ACCENT  = "#238636"
ACCENT2 = "#1f6feb"
FG      = "#e6edf3"
FG2     = "#8b949e"
BORDER  = "#30363d"
RED     = "#f85149"
YELLOW  = "#e3b341"

FONT_TITLE  = ("Consolas", 13, "bold")
FONT_LABEL  = ("Consolas", 9)
FONT_MONO   = ("Consolas", 8)
FONT_BUTTON = ("Consolas", 9, "bold")


def styled_frame(parent, **kw):
    return tk.Frame(parent, bg=BG2, relief="flat", **kw)


def label(parent, text, fg=FG, font=FONT_LABEL, **kw):
    return tk.Label(parent, text=text, bg=BG2, fg=fg, font=font, **kw)


def entry(parent, textvariable=None, width=18, **kw):
    e = tk.Entry(parent, textvariable=textvariable, width=width,
                 bg=BG3, fg=FG, insertbackground=FG,
                 relief="flat", bd=4, font=FONT_MONO, **kw)
    return e


def btn(parent, text, command, color=ACCENT, **kw):
    b = tk.Button(parent, text=text, command=command,
                  bg=color, fg="white", activebackground=color,
                  activeforeground="white", relief="flat",
                  font=FONT_BUTTON, cursor="hand2",
                  padx=10, pady=5, **kw)
    return b


def combo(parent, values, variable, width=14):
    c = ttk.Combobox(parent, values=values, textvariable=variable,
                     width=width, state="readonly", font=FONT_MONO)
    c.configure(style="Dark.TCombobox")
    return c


def separator(parent):
    return tk.Frame(parent, bg=BORDER, height=1)


# ═══════════════════════════════════════════════════════════════
# Pestaña SSB  — con TX por parlante y RX por micrófono
# ═══════════════════════════════════════════════════════════════

class SSBTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.mod_result   = None
        self.demod_result = None
        self.wav_path_1   = tk.StringVar(value="(ninguno)")
        self.wav_path_2   = tk.StringVar(value="(ninguno)")
        self._build()

    # ── Construcción de la interfaz ──────────────────────────────

    def _build(self):
        # ── Columna izquierda: parámetros compartidos ────────────
        ctrl = styled_frame(self)
        ctrl.pack(side="left", fill="y", padx=(8, 4), pady=8)

        label(ctrl, "⚡ SSB / ISB — HILBERT",
              fg=ACCENT2, font=FONT_TITLE).pack(pady=(12, 4))
        separator(ctrl).pack(fill="x", pady=4)

        # Parámetros comunes TX y RX
        params = [
            ("Frecuencia portadora [Hz]:", "fc",          "5000"),
            ("Error de fase [°] (RX):",    "phase_error", "0"),
            ("Error de frec. [Hz] (RX):",  "freq_error",  "0"),
        ]
        self.vars = {}
        for lbl_text, key, default in params:
            f = styled_frame(ctrl); f.pack(fill="x", padx=8, pady=3)
            label(f, lbl_text).pack(anchor="w")
            v = tk.StringVar(value=default)
            self.vars[key] = v
            entry(f, textvariable=v).pack(fill="x")

        f_mode = styled_frame(ctrl); f_mode.pack(fill="x", padx=8, pady=3)
        label(f_mode, "Tipo de modulación:").pack(anchor="w")
        self.mode_var = tk.StringVar(value="SSB-SC")
        combo(f_mode, ["SSB-SC", "SSB-FC", "ISB"], self.mode_var).pack(fill="x")

        f_sb = styled_frame(ctrl); f_sb.pack(fill="x", padx=8, pady=3)
        label(f_sb, "Banda lateral:").pack(anchor="w")
        self.sb_var = tk.StringVar(value="USB")
        combo(f_sb, ["USB", "LSB"], self.sb_var).pack(fill="x")

        separator(ctrl).pack(fill="x", pady=6)

        # Log
        label(ctrl, "Estado:", fg=FG2).pack(anchor="w", padx=8)
        self.log = tk.Text(ctrl, height=18, width=38, bg=BG3, fg=FG2,
                           font=FONT_MONO, relief="flat", state="disabled")
        self.log.pack(fill="x", padx=8, pady=4)

        # ── Columna central: TX ──────────────────────────────────
        tx_col = styled_frame(self)
        tx_col.pack(side="left", fill="y", padx=4, pady=8)

        label(tx_col, "📡  TRANSMISOR (TX)",
              fg=ACCENT, font=FONT_TITLE).pack(pady=(12, 4))
        separator(tx_col).pack(fill="x", pady=4)

        # WAV 1
        f1 = styled_frame(tx_col); f1.pack(fill="x", padx=8, pady=4)
        label(f1, "Archivo WAV (mensaje):").pack(anchor="w")
        r1 = tk.Frame(f1, bg=BG2); r1.pack(fill="x")
        self.lbl_wav1 = label(r1, "(ninguno)", fg=FG2, font=FONT_MONO)
        self.lbl_wav1.pack(side="left", expand=True, anchor="w")
        btn(r1, "📂", self._load_wav1, color=BG3).pack(side="right")

        # WAV 2 (ISB)
        f2 = styled_frame(tx_col); f2.pack(fill="x", padx=8, pady=2)
        label(f2, "WAV 2 — banda inferior (solo ISB):").pack(anchor="w")
        r2 = tk.Frame(f2, bg=BG2); r2.pack(fill="x")
        self.lbl_wav2 = label(r2, "(ninguno)", fg=FG2, font=FONT_MONO)
        self.lbl_wav2.pack(side="left", expand=True, anchor="w")
        btn(r2, "📂", self._load_wav2, color=BG3).pack(side="right")

        separator(tx_col).pack(fill="x", pady=6)

        btn(tx_col, "▶  MODULAR (ver gráficas)",
            self._run_modulate, ACCENT).pack(fill="x", padx=8, pady=3)

        btn(tx_col, "📡  MODULAR + TRANSMITIR por parlante",
            self._run_tx, "#2ea043").pack(fill="x", padx=8, pady=3)

        separator(tx_col).pack(fill="x", pady=4)

        label(tx_col, "Opciones de reproducción:", fg=FG2).pack(anchor="w", padx=8)
        btn(tx_col, "🔊  Reproducir mensaje original",
            lambda: self._play_key("message"), BG3).pack(fill="x", padx=8, pady=2)
        btn(tx_col, "🔊  Reproducir señal SSB modulada",
            lambda: self._play_key("ssb"), BG3).pack(fill="x", padx=8, pady=2)
        btn(tx_col, "💾  Guardar SSB como WAV",
            self._save_ssb_wav, BG3).pack(fill="x", padx=8, pady=2)

        # ── Columna derecha: RX ──────────────────────────────────
        rx_col = styled_frame(self)
        rx_col.pack(side="left", fill="y", padx=4, pady=8)

        label(rx_col, "🎙  RECEPTOR (RX)",
              fg=YELLOW, font=FONT_TITLE).pack(pady=(12, 4))
        separator(rx_col).pack(fill="x", pady=4)

        # Duración de grabación
        f_dur = styled_frame(rx_col); f_dur.pack(fill="x", padx=8, pady=4)
        label(f_dur, "Duración grabación [s]:").pack(anchor="w")
        self.dur_var = tk.StringVar(value="10")
        entry(f_dur, textvariable=self.dur_var).pack(fill="x")

        label(rx_col,
              "El RX graba por micrófono\ny demodula lo que escucha.\nUsá la misma fc\nque el TX.",
              fg=FG2, font=FONT_MONO).pack(padx=8, pady=4, anchor="w")

        separator(rx_col).pack(fill="x", pady=6)

        btn(rx_col, "🎙  GRABAR + DEMODULAR",
            self._run_rx, YELLOW).pack(fill="x", padx=8, pady=3)

        separator(rx_col).pack(fill="x", pady=4)

        label(rx_col, "Después de recibir:", fg=FG2).pack(anchor="w", padx=8)
        btn(rx_col, "🔊  Reproducir señal recibida (SSB)",
            lambda: self._play_rx("ssb_rx"), BG3).pack(fill="x", padx=8, pady=2)
        btn(rx_col, "🔊  Reproducir mensaje recuperado",
            lambda: self._play_rx("recovered"), BG3).pack(fill="x", padx=8, pady=2)
        btn(rx_col, "💾  Guardar mensaje recuperado",
            self._save_wav, BG3).pack(fill="x", padx=8, pady=2)

        # ── Panel de gráficas (derecha completa) ─────────────────
        self.plot_frame = tk.Frame(self, bg=BG)
        self.plot_frame.pack(side="left", fill="both", expand=True,
                             padx=(4, 8), pady=8)
        self.canvas = None

    # ── Helpers ──────────────────────────────────────────────────

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _load_wav1(self):
        p = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if p:
            self.wav_path_1.set(p)
            self.lbl_wav1.config(text=os.path.basename(p), fg=FG)
            self._log(f"WAV 1: {os.path.basename(p)}")

    def _load_wav2(self):
        p = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if p:
            self.wav_path_2.set(p)
            self.lbl_wav2.config(text=os.path.basename(p), fg=FG)
            self._log(f"WAV 2: {os.path.basename(p)}")

    def _get_params(self):
        try:
            return (float(self.vars["fc"].get()),
                    float(self.vars["phase_error"].get()),
                    float(self.vars["freq_error"].get()))
        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos.")
            return None

    def _show_figure(self, fig):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_mod_result(self, fc, mode, sb):
        """Carga WAVs y genera mod_result. Retorna True si tuvo éxito."""
        if self.wav_path_1.get() == "(ninguno)":
            self.after(0, lambda: messagebox.showwarning(
                "Advertencia", "Cargue el archivo WAV del mensaje."))
            return False
        msg1, fs = load_wav(self.wav_path_1.get())
        if mode == "ISB":
            if self.wav_path_2.get() == "(ninguno)":
                self.after(0, lambda: messagebox.showwarning(
                    "Advertencia", "ISB requiere dos archivos WAV."))
                return False
            msg2, _ = load_wav(self.wav_path_2.get())
            r = modulate_isb(msg1, msg2, fs, fc)
            r.update({"mode": "ISB", "message": msg1,
                      "ssb": r["isb"], "dsb": r["isb"], "sideband": "USB+LSB"})
        else:
            r = modulate_ssb(msg1, fs, fc, mode=mode, sideband=sb)
        self.mod_result = r
        return True

    # ── Acciones TX ──────────────────────────────────────────────

    def _run_modulate(self):
        """Solo genera y muestra gráficas — sin reproducir."""
        p = self._get_params()
        if not p: return
        fc, ph, fe = p
        mode, sb = self.mode_var.get(), self.sb_var.get()

        def task():
            try:
                self._log("Modulando...")
                if not self._build_mod_result(fc, mode, sb): return
                self._log(f"✓ Modulación {mode} lista.")
                fig = plot_ssb_modulator(self.mod_result)
                self.after(0, lambda: self._show_figure(fig))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=task, daemon=True).start()

    def _run_tx(self):
        """Modula y transmite la señal SSB por el parlante."""
        p = self._get_params()
        if not p: return
        fc, ph, fe = p
        mode, sb = self.mode_var.get(), self.sb_var.get()

        def task():
            try:
                self._log("Preparando modulación para TX...")
                if not self._build_mod_result(fc, mode, sb): return
                self._log(f"✓ {mode} modulada. Transmitiendo por parlante...")
                fig = plot_ssb_modulator(self.mod_result)
                self.after(0, lambda: self._show_figure(fig))
                # Reproducir señal modulada (esto bloquea hasta que termina)
                dur = transmit_ssb(self.mod_result, progress_cb=self._log)
                self._log(f"✓ TX completo — {dur:.1f}s transmitidos.")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error TX", str(e)))
        threading.Thread(target=task, daemon=True).start()

    def _play_key(self, key):
        """Reproduce una señal del mod_result por clave."""
        if self.mod_result is None:
            messagebox.showwarning("Advertencia", "Primero modulá una señal.")
            return
        audio = self.mod_result.get(key)
        if audio is None:
            messagebox.showwarning("Advertencia", f"No hay señal '{key}'.")
            return
        fs = self.mod_result["fs"]
        def play():
            self._log(f"🔊 Reproduciendo {key}...")
            sd.play(audio.astype(np.float32), samplerate=fs)
            sd.wait()
            self._log("✓ Listo.")
        threading.Thread(target=play, daemon=True).start()

    def _save_ssb_wav(self):
        if self.mod_result is None:
            messagebox.showwarning("Advertencia", "Primero modulá.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")])
        if path:
            save_wav(path, self.mod_result["ssb"], self.mod_result["fs"])
            self._log(f"💾 SSB guardada: {os.path.basename(path)}")

    # ── Acciones RX ──────────────────────────────────────────────

    def _run_rx(self):
        """Graba por micrófono y demodula la señal SSB recibida."""
        p = self._get_params()
        if not p: return
        fc, ph, fe = p
        sb = self.sb_var.get()
        try:
            dur = float(self.dur_var.get())
        except ValueError:
            messagebox.showerror("Error", "Duración inválida.")
            return

        # Para las gráficas necesitamos un mod_result de referencia.
        # Si no hay uno (equipo receptor no moduló), creamos uno ficticio
        # con la señal recibida como referencia.
        def task():
            try:
                self.demod_result = receive_ssb(
                    fs=44100, fc=fc, duration_s=dur,
                    sideband=sb,
                    phase_error_deg=ph,
                    freq_error_hz=fe,
                    progress_cb=self._log
                )
                self._log(f"✓ RX completo. Demodulando...")

                # Si no hay mod_result (solo RX), usamos ssb_rx como referencia
                if self.mod_result is None:
                    ref = {
                        "message": self.demod_result["ssb_rx"],
                        "dsb":     self.demod_result["ssb_rx"],
                        "ssb":     self.demod_result["ssb_rx"],
                        "t":       self.demod_result["t"],
                        "fs":      self.demod_result["fs"],
                        "fc":      fc,
                        "mode":    "SSB-SC",
                        "sideband": sb,
                    }
                else:
                    ref = self.mod_result

                fig = plot_ssb_demodulator(ref, self.demod_result)
                self.after(0, lambda: self._show_figure(fig))
                self._log("✓ Gráficas RX generadas.")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error RX", str(e)))
        threading.Thread(target=task, daemon=True).start()

    def _play_rx(self, key):
        """Reproduce una señal del demod_result."""
        if self.demod_result is None:
            messagebox.showwarning("Advertencia", "Primero realizá la recepción (RX).")
            return
        audio = self.demod_result.get(key)
        if audio is None:
            messagebox.showwarning("Advertencia", f"No hay señal '{key}'.")
            return
        fs = self.demod_result["fs"]
        def play():
            self._log(f"🔊 Reproduciendo {key}...")
            sd.play(audio.astype(np.float32), samplerate=fs)
            sd.wait()
            self._log("✓ Listo.")
        threading.Thread(target=play, daemon=True).start()

    def _save_wav(self):
        if self.demod_result is None:
            messagebox.showwarning("Advertencia", "No hay señal demodulada (RX).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")])
        if path:
            save_wav(path, self.demod_result["recovered"], self.demod_result["fs"])
            self._log(f"💾 Mensaje recuperado guardado: {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════
# Pestaña Digital Pasobanda
# ═══════════════════════════════════════════════════════════════

class DigitalTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.file_path = tk.StringVar(value="(ninguno)")
        self.use_fec   = tk.BooleanVar(value=True)
        self.rx_audio  = None
        self.rx_data   = None
        self.rx_result = None
        self.tx_audio  = None
        self._build()

    def _build(self):
        # ── Panel de control con scroll ─────────────────────────
        ctrl_outer = tk.Frame(self, bg=BG2)
        ctrl_outer.pack(side="left", fill="y", padx=(8,4), pady=8)

        ctrl_canvas = tk.Canvas(ctrl_outer, bg=BG2, highlightthickness=0, width=260)
        ctrl_canvas.pack(side="left", fill="y", expand=True)

        ctrl_sb = tk.Scrollbar(ctrl_outer, orient="vertical",
                               command=ctrl_canvas.yview)
        ctrl_sb.pack(side="right", fill="y")
        ctrl_canvas.configure(yscrollcommand=ctrl_sb.set)

        ctrl = tk.Frame(ctrl_canvas, bg=BG2)
        ctrl_win = ctrl_canvas.create_window((0, 0), window=ctrl, anchor="nw")

        def _on_frame_configure(e):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
        ctrl.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(e):
            ctrl_canvas.itemconfig(ctrl_win, width=e.width)
        ctrl_canvas.bind("<Configure>", _on_canvas_configure)

        # Scroll con rueda del mouse
        def _on_mousewheel(e):
            ctrl_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        label(ctrl, "📡 MODULACIÓN DIGITAL",
              fg=YELLOW, font=FONT_TITLE).pack(pady=(12,4))
        separator(ctrl).pack(fill="x", pady=4)

        # Parámetros del sistema (info)
        info = styled_frame(ctrl); info.pack(fill="x", padx=8, pady=4)
        for txt in [f"Modulación : BFSK",
                    f"Mark (1)   : {F_MARK} Hz",
                    f"Space (0)  : {F_SPACE} Hz",
                    f"Baud rate  : 300 Bd",
                    f"Fs         : {FS} Hz"]:
            label(info, txt, fg=FG2, font=FONT_MONO).pack(anchor="w")

        separator(ctrl).pack(fill="x", pady=4)

        # Archivo
        f_file = styled_frame(ctrl); f_file.pack(fill="x", padx=8, pady=4)
        label(f_file, "Archivo a transmitir:").pack(anchor="w")
        row = tk.Frame(f_file, bg=BG2); row.pack(fill="x")
        self.lbl_file = label(row, "(ninguno)", fg=FG2, font=FONT_MONO)
        self.lbl_file.pack(side="left", expand=True, anchor="w")
        btn(row, "📂", self._load_file, color=BG3).pack(side="right")

        # FEC
        f_fec = styled_frame(ctrl); f_fec.pack(fill="x", padx=8, pady=4)
        tk.Checkbutton(f_fec, text="Usar corrección de errores (Hamming 7,4)",
                       variable=self.use_fec, bg=BG2, fg=FG, selectcolor=BG3,
                       activebackground=BG2, activeforeground=FG,
                       font=FONT_LABEL).pack(anchor="w")

        # Duración RX
        f_dur = styled_frame(ctrl); f_dur.pack(fill="x", padx=8, pady=4)
        label(f_dur, "Duración grabación RX [s]:").pack(anchor="w")
        self.dur_var = tk.StringVar(value="30")
        entry(f_dur, textvariable=self.dur_var).pack(fill="x")

        separator(ctrl).pack(fill="x", pady=6)

        btn(ctrl, "▶  TRANSMITIR (TX)", self._run_tx,  ACCENT).pack(fill="x", padx=8, pady=3)
        btn(ctrl, "💾  GUARDAR AUDIO TX (.wav)", self._save_tx_wav, BG3).pack(fill="x", padx=8, pady=3)

        separator(ctrl).pack(fill="x", pady=4)

        # ── Control de ruido AWGN ────────────────────────────────
        f_snr = styled_frame(ctrl); f_snr.pack(fill="x", padx=8, pady=4)
        snr_row = tk.Frame(f_snr, bg=BG2); snr_row.pack(fill="x")
        label(snr_row, "SNR canal [dB]:", fg=FG2).pack(side="left")
        self.snr_val_lbl = label(snr_row, "20 dB", fg=ACCENT, font=FONT_MONO)
        self.snr_val_lbl.pack(side="right")
        self.snr_var = tk.DoubleVar(value=20.0)
        snr_scale = tk.Scale(f_snr, from_=0, to=40, resolution=1,
                             orient="horizontal", variable=self.snr_var,
                             bg=BG2, fg=FG2, troughcolor=BG3,
                             highlightthickness=0, showvalue=False,
                             command=lambda v: self.snr_val_lbl.config(
                                 text=f"{int(float(v))} dB"))
        snr_scale.pack(fill="x", pady=2)
        btn(ctrl, "🔊  GUARDAR WAV CON RUIDO AWGN", self._save_noisy_wav, BG3).pack(fill="x", padx=8, pady=3)

        separator(ctrl).pack(fill="x", pady=4)

        btn(ctrl, "🎙  RECIBIR   (RX)", self._run_rx,  ACCENT2).pack(fill="x", padx=8, pady=3)
        btn(ctrl, "📂  DEMODULAR DESDE ARCHIVO", self._run_rx_file, ACCENT2).pack(fill="x", padx=8, pady=3)
        btn(ctrl, "📊  SIMULACIÓN BER", self._run_ber, BG3).pack(fill="x", padx=8, pady=3)

        separator(ctrl).pack(fill="x", pady=6)

        # Resultado RX
        self.rx_result_lbl = label(ctrl, "", fg=FG2, font=FONT_MONO)
        self.rx_result_lbl.pack(padx=8, anchor="w")

        btn(ctrl, "💾  GUARDAR ARCHIVO RX", self._save_rx, BG3).pack(fill="x", padx=8, pady=3)

        # Log
        label(ctrl, "Estado:", fg=FG2).pack(anchor="w", padx=8)
        self.log = tk.Text(ctrl, height=18, width=38, bg=BG3, fg=FG2,
                           font=FONT_MONO, relief="flat", state="disabled")
        self.log.pack(fill="x", padx=8, pady=4)

        # ── Gráficas ─────────────────────────────────────────────
        self.plot_frame = tk.Frame(self, bg=BG)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=(4,8), pady=8)
        self.canvas = None

    # ── Helpers ─────────────────────────────────────────────────

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _show_figure(self, fig):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _load_file(self):
        p = filedialog.askopenfilename()
        if p:
            self.file_path.set(p)
            self.lbl_file.config(text=os.path.basename(p), fg=FG)
            size_kb = os.path.getsize(p) / 1024
            self._log(f"Archivo: {os.path.basename(p)} ({size_kb:.1f} KB)")

    # ── Acciones ────────────────────────────────────────────────

    def _run_tx(self):
        if self.file_path.get() == "(ninguno)":
            messagebox.showwarning("Advertencia", "Seleccione un archivo.")
            return

        def task():
            self._log("Construyendo trama...")
            result = transmit(self.file_path.get(),
                              use_fec=self.use_fec.get(),
                              progress_cb=self._log)
            self.tx_audio = result["audio"]
            self._log(f"✓ Transmisión completa: {result['bytes_sent']} bytes")
            self._log(f"  Duración: {result['duration_s']:.1f}s")
            self._log(f"  Tasa: {result['baud_rate']:.0f} bps efectivos")
            self._log(f"  [DEBUG] tx_audio shape: {self.tx_audio.shape}, dtype: {self.tx_audio.dtype}, max: {np.max(np.abs(self.tx_audio)):.4f}")
            fig = plot_digital_tx(result["audio"], FS, F_MARK, F_SPACE)
            self.after(0, lambda: self._show_figure(fig))

        threading.Thread(target=task, daemon=True).start()

    def _run_rx(self):
        try:
            dur = float(self.dur_var.get())
        except ValueError:
            messagebox.showerror("Error", "Duración inválida.")
            return

        def task():
            self._log(f"🎙 Grabando {dur}s...")
            result = receive(dur, progress_cb=self._log)
            self.rx_audio   = result.get("audio")
            self.rx_data    = result.get("data")
            self.rx_result  = result

            if result.get("success"):
                self.after(0, lambda: self.rx_result_lbl.config(
                    text=f"✓ {result['length']} bytes | "
                         f"{result['corrections']} correcciones",
                    fg=ACCENT))
                self._log(f"✓ Recepción OK: {result['length']} bytes")
            else:
                self.after(0, lambda: self.rx_result_lbl.config(
                    text="✗ Error de recepción", fg=RED))
                self._log("✗ Error en checksum o trama corrupta.")

        threading.Thread(target=task, daemon=True).start()

    def _run_ber(self):
        def task():
            self._log("Calculando curva BER...")
            snr = np.linspace(-2, 14, 30)
            ber_data = compute_ber_curve(snr)

            # Usar audio de TX o generar señal de prueba
            audio = self.tx_audio if self.tx_audio is not None else \
                    np.sin(2*np.pi*F_MARK*np.arange(FS)//FS)

            fig = plot_ber_and_eye(ber_data, audio, FS, SPB)
            self._log("✓ BER y diagrama de ojo generados.")
            self.after(0, lambda: self._show_figure(fig))

        threading.Thread(target=task, daemon=True).start()

    def _save_rx(self):
        if self.rx_data is None:
            messagebox.showwarning("Advertencia",
                "No hay datos recibidos aún.\n"
                "Usá 'RECIBIR (RX)' o 'DEMODULAR DESDE ARCHIVO' primero.")
            return
        path = filedialog.asksaveasfilename()
        if path:
            with open(path, 'wb') as f:
                f.write(self.rx_data)
            self._log(f"💾 Guardado: {os.path.basename(path)}")

    def _save_tx_wav(self):
        """Guarda el audio BFSK modulado como archivo WAV."""
        if self.tx_audio is None:
            messagebox.showwarning("Advertencia", "Primero ejecutá TX para generar el audio.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            title="Guardar audio modulado BFSK")
        if path:
            save_modulated_wav(path, self.tx_audio, FS)
            self._log(f"💾 Audio TX guardado: {os.path.basename(path)}")

    def _save_noisy_wav(self):
        """Agrega ruido AWGN al audio TX y lo guarda como WAV."""
        if self.tx_audio is None:
            messagebox.showwarning("Advertencia", "Primero ejecutá TX para generar el audio.")
            return
        snr_db = self.snr_var.get()
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            title=f"Guardar audio con ruido AWGN ({int(snr_db)} dB)")
        if not path:
            return
        noisy = add_awgn(self.tx_audio, snr_db)
        save_modulated_wav(path, noisy, FS)
        # Calcular SNR real del audio resultante
        signal_power = np.mean(self.tx_audio ** 2)
        noise        = noisy - self.tx_audio
        noise_power  = np.mean(noise ** 2)
        snr_real     = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        self._log(f"🔊 Audio con ruido guardado: {os.path.basename(path)}")
        self._log(f"   SNR solicitado: {int(snr_db)} dB | SNR real: {snr_real:.1f} dB")

    def _run_rx_file(self):
        """Demodula desde un archivo WAV (sin usar micrófono)."""
        path = filedialog.askopenfilename(
            filetypes=[("WAV audio", "*.wav")],
            title="Abrir audio BFSK modulado")
        if not path:
            return

        def task():
            try:
                self._log(f"📂 Demodulando desde archivo: {os.path.basename(path)}")
                result = receive_from_file(path, progress_cb=self._log)
                self.rx_audio  = result.get("audio")
                self.rx_data   = result.get("data")
                self.rx_result = result

                if result.get("success"):
                    self.after(0, lambda: self.rx_result_lbl.config(
                        text=f"✓ {result['length']} bytes | "
                             f"{result['corrections']} correcciones",
                        fg=ACCENT))
                    self._log(f"✓ Demodulación OK: {result['length']} bytes, "
                              f"{result['corrections']} bits corregidos")
                    fig = plot_digital_tx(self.rx_audio, result["fs"], F_MARK, F_SPACE)
                    self.after(0, lambda: self._show_figure(fig))
                else:
                    self.after(0, lambda: self.rx_result_lbl.config(
                        text="✗ Error de demodulación", fg=RED))
                    self._log(f"✗ Error: {result.get('error', 'desconocido')}")
            except Exception as e:
                import traceback
                self._log(f"💥 Excepción en RX: {e}")
                self._log(traceback.format_exc())

        threading.Thread(target=task, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
# Ventana principal
# ═══════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EL5522 — Simulador de Comunicaciones Eléctricas")
        self.configure(bg=BG)
        self.geometry("1280x760")
        self.resizable(True, True)
        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",           background=BG,  borderwidth=0)
        style.configure("TNotebook.Tab",       background=BG3, foreground=FG2,
                        padding=(14, 6), font=FONT_BUTTON)
        style.map("TNotebook.Tab",
                  background=[("selected", BG2)],
                  foreground=[("selected", FG)])
        style.configure("Dark.TCombobox",
                        fieldbackground=BG3, background=BG3,
                        foreground=FG, arrowcolor=FG,
                        selectbackground=BG3, selectforeground=FG)

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG, pady=8)
        header.pack(fill="x", padx=12)
        tk.Label(header,
                 text="⚡ TALLER DE COMUNICACIONES ELÉCTRICAS — EL5522",
                 bg=BG, fg=ACCENT2, font=("Consolas", 14, "bold")).pack(side="left")
        tk.Label(header,
                 text="Modulación SSB + Digital Pasobanda",
                 bg=BG, fg=FG2, font=FONT_LABEL).pack(side="left", padx=(14,0))

        # Separador
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Notebook (pestañas)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_ssb  = SSBTab(nb)
        self.tab_dig  = DigitalTab(nb)

        nb.add(self.tab_ssb, text="  📻  SSB / ISB (Hilbert)  ")
        nb.add(self.tab_dig, text="  📡  Digital Pasobanda  ")

        # Footer
        footer = tk.Frame(self, bg=BG3, pady=3)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer,
                 text="EL5522 · Instituto Tecnológico de Costa Rica · 2026",
                 bg=BG3, fg=FG2, font=FONT_MONO).pack()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()