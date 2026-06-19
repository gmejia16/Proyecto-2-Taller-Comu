"""
Módulo Principal de Interfaz Gráfica (GUI) - App.py

Punto de entrada del Simulador de Comunicaciones Eléctricas.
Implementa una arquitectura de diseño basada en eventos utilizando Tkinter.
Integra los módulos de procesamiento digital de señales (core) y visualización (utils)
operando sobre hilos independientes (threading) para evitar el bloqueo del bucle 
principal de la interfaz durante las operaciones I/O del hardware acústico.

Curso: EL5522 - Taller de Comunicaciones Eléctricas
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

# Integración al path del sistema para resolución absoluta de módulos
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

# =============================================================================
# Paleta de Colores y Estilos Tipográficos (Estilo Dark Theme)
# =============================================================================

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

# =============================================================================
# Funciones Constructor de Interfaz (UI Builders)
# =============================================================================

def styled_frame(parent, **kw) -> tk.Frame:
    """Genera un contenedor de diseño estandarizado."""
    return tk.Frame(parent, bg=BG2, relief="flat", **kw)

def label(parent, text, fg=FG, font=FONT_LABEL, **kw) -> tk.Label:
    """Genera una etiqueta de texto estandarizada."""
    return tk.Label(parent, text=text, bg=BG2, fg=fg, font=font, **kw)

def entry(parent, textvariable=None, width=18, **kw) -> tk.Entry:
    """Genera un campo de entrada de texto estandarizado."""
    return tk.Entry(parent, textvariable=textvariable, width=width,
                 bg=BG3, fg=FG, insertbackground=FG,
                 relief="flat", bd=4, font=FONT_MONO, **kw)

def btn(parent, text, command, color=ACCENT, **kw) -> tk.Button:
    """Genera un botón interactivo estandarizado."""
    return tk.Button(parent, text=text, command=command,
                  bg=color, fg="white", activebackground=color,
                  activeforeground="white", relief="flat",
                  font=FONT_BUTTON, cursor="hand2",
                  padx=10, pady=5, **kw)

def combo(parent, values, variable, width=14) -> ttk.Combobox:
    """Genera un menú desplegable de selección estandarizado."""
    c = ttk.Combobox(parent, values=values, textvariable=variable,
                     width=width, state="readonly", font=FONT_MONO)
    c.configure(style="Dark.TCombobox")
    return c

def separator(parent) -> tk.Frame:
    """Genera una línea divisoria visual."""
    return tk.Frame(parent, bg=BORDER, height=1)

# =============================================================================
# Controlador del Transceptor Analógico SSB/ISB
# =============================================================================

class SSBTab(tk.Frame):
    """
    Clase controladora para la pestaña de modulación analógica.
    Gestiona la configuración de portadoras, carga de archivos de audio
    y la interconexión con el núcleo matemático de la Transformada de Hilbert.
    """
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.mod_result   = None
        self.demod_result = None
        self.wav_path_1   = tk.StringVar(value="(ninguno)")
        self.wav_path_2   = tk.StringVar(value="(ninguno)")
        self._build()

    def _build(self):
        """Construye la jerarquía visual de la pestaña SSB."""
        # ── Columna de Parámetros Globales ──
        ctrl = styled_frame(self)
        ctrl.pack(side="left", fill="y", padx=(8, 4), pady=8)

        label(ctrl, "SSB / ISB — HILBERT",
              fg=ACCENT2, font=FONT_TITLE).pack(pady=(12, 4))
        separator(ctrl).pack(fill="x", pady=4)

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
        label(ctrl, "Registro de Eventos:", fg=FG2).pack(anchor="w", padx=8)
        self.log = tk.Text(ctrl, height=18, width=38, bg=BG3, fg=FG2,
                           font=FONT_MONO, relief="flat", state="disabled")
        self.log.pack(fill="x", padx=8, pady=4)

        # ── Columna Transmisor (TX) ──
        tx_col = styled_frame(self)
        tx_col.pack(side="left", fill="y", padx=4, pady=8)

        label(tx_col, "TRANSMISOR (TX)", fg=ACCENT, font=FONT_TITLE).pack(pady=(12, 4))
        separator(tx_col).pack(fill="x", pady=4)

        f1 = styled_frame(tx_col); f1.pack(fill="x", padx=8, pady=4)
        label(f1, "Archivo WAV (Banda Base):").pack(anchor="w")
        r1 = tk.Frame(f1, bg=BG2); r1.pack(fill="x")
        self.lbl_wav1 = label(r1, "(ninguno)", fg=FG2, font=FONT_MONO)
        self.lbl_wav1.pack(side="left", expand=True, anchor="w")
        btn(r1, "Examinar", self._load_wav1, color=BG3).pack(side="right")

        f2 = styled_frame(tx_col); f2.pack(fill="x", padx=8, pady=2)
        label(f2, "WAV LSB (Modo ISB):").pack(anchor="w")
        r2 = tk.Frame(f2, bg=BG2); r2.pack(fill="x")
        self.lbl_wav2 = label(r2, "(ninguno)", fg=FG2, font=FONT_MONO)
        self.lbl_wav2.pack(side="left", expand=True, anchor="w")
        btn(r2, "Examinar", self._load_wav2, color=BG3).pack(side="right")

        separator(tx_col).pack(fill="x", pady=6)
        btn(tx_col, "Ejecutar Modulación Analítica", self._run_modulate, ACCENT).pack(fill="x", padx=8, pady=3)
        btn(tx_col, "Transmitir Físicamente (Parlante)", self._run_tx, "#2ea043").pack(fill="x", padx=8, pady=3)

        separator(tx_col).pack(fill="x", pady=4)
        label(tx_col, "Herramientas:", fg=FG2).pack(anchor="w", padx=8)
        btn(tx_col, "Reproducir Banda Base Original", lambda: self._play_key("message"), BG3).pack(fill="x", padx=8, pady=2)
        btn(tx_col, "Reproducir Señal Modulada (SSB)", lambda: self._play_key("ssb"), BG3).pack(fill="x", padx=8, pady=2)
        btn(tx_col, "Exportar SSB a WAV", self._save_ssb_wav, BG3).pack(fill="x", padx=8, pady=2)

        # ── Columna Receptor (RX) ──
        rx_col = styled_frame(self)
        rx_col.pack(side="left", fill="y", padx=4, pady=8)

        label(rx_col, "RECEPTOR (RX)", fg=YELLOW, font=FONT_TITLE).pack(pady=(12, 4))
        separator(rx_col).pack(fill="x", pady=4)

        f_dur = styled_frame(rx_col); f_dur.pack(fill="x", padx=8, pady=4)
        label(f_dur, "Ventana de Captura [s]:").pack(anchor="w")
        self.dur_var = tk.StringVar(value="10")
        entry(f_dur, textvariable=self.dur_var).pack(fill="x")

        label(rx_col, "El receptor ejecuta captura\nmicrófonica y demodulación\nsíncrona en tiempo real.",
              fg=FG2, font=FONT_MONO).pack(padx=8, pady=4, anchor="w")

        separator(rx_col).pack(fill="x", pady=6)
        btn(rx_col, "Iniciar Escucha y Demodulación", self._run_rx, YELLOW).pack(fill="x", padx=8, pady=3)

        separator(rx_col).pack(fill="x", pady=4)
        label(rx_col, "Resultados de Recepción:", fg=FG2).pack(anchor="w", padx=8)
        btn(rx_col, "Reproducir Captura de Canal (RX)", lambda: self._play_rx("ssb_rx"), BG3).pack(fill="x", padx=8, pady=2)
        btn(rx_col, "Reproducir Banda Base Recuperada", lambda: self._play_rx("recovered"), BG3).pack(fill="x", padx=8, pady=2)
        btn(rx_col, "Exportar Banda Base a WAV", self._save_wav, BG3).pack(fill="x", padx=8, pady=2)

        # ── Contenedor de Gráficas (Lienzo Matplotlib) ──
        self.plot_frame = tk.Frame(self, bg=BG)
        self.plot_frame.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
        self.canvas = None

    def _log(self, msg: str):
        """Añade una entrada al registro de eventos de la GUI."""
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _load_wav1(self):
        p = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
        if p:
            self.wav_path_1.set(p)
            self.lbl_wav1.config(text=os.path.basename(p), fg=FG)
            self._log(f"Asignado WAV 1: {os.path.basename(p)}")

    def _load_wav2(self):
        p = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
        if p:
            self.wav_path_2.set(p)
            self.lbl_wav2.config(text=os.path.basename(p), fg=FG)
            self._log(f"Asignado WAV 2: {os.path.basename(p)}")

    def _get_params(self) -> tuple:
        """Valida y extrae los parámetros numéricos de la interfaz."""
        try:
            return (float(self.vars["fc"].get()),
                    float(self.vars["phase_error"].get()),
                    float(self.vars["freq_error"].get()))
        except ValueError:
            messagebox.showerror("Error de Formato", "Los parámetros de frecuencia y fase deben ser numéricos.")
            return None

    def _show_figure(self, fig):
        """Incrusta de forma segura una figura de Matplotlib en el lienzo de Tkinter."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_mod_result(self, fc, mode, sb) -> bool:
        """Carga la memoria temporal con el procesamiento analógico previo a la transmisión."""
        if self.wav_path_1.get() == "(ninguno)":
            self.after(0, lambda: messagebox.showwarning("Advertencia", "Asigne un archivo de banda base."))
            return False
            
        msg1, fs = load_wav(self.wav_path_1.get())
        if mode == "ISB":
            if self.wav_path_2.get() == "(ninguno)":
                self.after(0, lambda: messagebox.showwarning("Advertencia", "El modo ISB requiere dos fuentes de audio."))
                return False
            msg2, _ = load_wav(self.wav_path_2.get())
            r = modulate_isb(msg1, msg2, fs, fc)
            r.update({"mode": "ISB", "message": msg1, "ssb": r["isb"], "dsb": r["isb"], "sideband": "USB+LSB"})
        else:
            r = modulate_ssb(msg1, fs, fc, mode=mode, sideband=sb)
        self.mod_result = r
        return True

    def _run_modulate(self):
        """
        Ejecuta la modulación en un hilo secundario para prevenir bloqueos de la GUI.
        Renderiza las gráficas sin activar hardware de audio.
        """
        p = self._get_params()
        if not p: return
        fc, ph, fe = p
        mode, sb = self.mode_var.get(), self.sb_var.get()

        def task():
            try:
                self._log("Iniciando procesamiento analítico...")
                if not self._build_mod_result(fc, mode, sb): return
                self._log(f"✓ Modulación {mode} completada.")
                fig = plot_ssb_modulator(self.mod_result)
                self.after(0, lambda: self._show_figure(fig))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Excepción DSP", str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def _run_tx(self):
        """Ejecuta modulación y transmisión acústica bloqueante sobre un hilo asíncrono."""
        p = self._get_params()
        if not p: return
        fc, ph, fe = p
        mode, sb = self.mode_var.get(), self.sb_var.get()

        def task():
            try:
                self._log("Acondicionando hardware para TX...")
                if not self._build_mod_result(fc, mode, sb): return
                self._log(f"✓ {mode} sintetizada. Abriendo stream de audio...")
                fig = plot_ssb_modulator(self.mod_result)
                self.after(0, lambda: self._show_figure(fig))
                
                dur = transmit_ssb(self.mod_result, progress_cb=self._log)
                self._log(f"✓ TX Físico completado ({dur:.1f}s).")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Fallo de Hardware", str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def _play_key(self, key):
        """Despachador de reproducción asíncrona de señales temporales."""
        if self.mod_result is None:
            messagebox.showwarning("Aviso", "Memoria vacía. Ejecute modulación previamente.")
            return
        audio = self.mod_result.get(key)
        if audio is None: return
        fs = self.mod_result["fs"]
        
        def play():
            self._log(f"Reproduciendo vector '{key}'...")
            sd.play(audio.astype(np.float32), samplerate=fs)
            sd.wait()
            self._log("✓ Stream cerrado.")
            
        threading.Thread(target=play, daemon=True).start()

    def _save_ssb_wav(self):
        if self.mod_result is None: return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV PCM", "*.wav")])
        if path:
            save_wav(path, self.mod_result["ssb"], self.mod_result["fs"])
            self._log(f"✓ Exportación I/Q completada: {os.path.basename(path)}")

    def _run_rx(self):
        """Inicia captura de micrófono y aplica filtro de demodulación síncrona."""
        p = self._get_params()
        if not p: return
        fc, ph, fe = p
        sb = self.sb_var.get()
        try:
            dur = float(self.dur_var.get())
        except ValueError:
            messagebox.showerror("Error", "Tiempo de captura inválido.")
            return

        def task():
            try:
                self.demod_result = receive_ssb(
                    fs=44100, fc=fc, duration_s=dur,
                    sideband=sb, phase_error_deg=ph,
                    freq_error_hz=fe, progress_cb=self._log
                )
                self._log(f"✓ Buffer RX lleno. Aplicando filtro demodulador...")

                # Simulación de referencia si no existe TX previo en la misma máquina
                if self.mod_result is None:
                    ref = {
                        "message": self.demod_result["ssb_rx"],
                        "dsb":     self.demod_result["ssb_rx"],
                        "ssb":     self.demod_result["ssb_rx"],
                        "t":       self.demod_result["t"],
                        "fs":      self.demod_result["fs"],
                        "fc":      fc, "mode": "SSB-SC", "sideband": sb,
                    }
                else:
                    ref = self.mod_result

                fig = plot_ssb_demodulator(ref, self.demod_result)
                self.after(0, lambda: self._show_figure(fig))
                self._log("✓ Extracción de banda base completada.")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Excepción de Interrupción RX", str(e)))
                
        threading.Thread(target=task, daemon=True).start()

    def _play_rx(self, key):
        if self.demod_result is None: return
        audio = self.demod_result.get(key)
        if audio is None: return
        fs = self.demod_result["fs"]
        
        def play():
            self._log(f"Reproduciendo vector '{key}'...")
            sd.play(audio.astype(np.float32), samplerate=fs)
            sd.wait()
            
        threading.Thread(target=play, daemon=True).start()

    def _save_wav(self):
        if self.demod_result is None: return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV PCM", "*.wav")])
        if path:
            save_wav(path, self.demod_result["recovered"], self.demod_result["fs"])
            self._log(f"✓ Banda base recuperada exportada: {os.path.basename(path)}")


# =============================================================================
# Controlador del Transceptor Digital (Módem BFSK)
# =============================================================================

class DigitalTab(tk.Frame):
    """
    Clase controladora para el módem digital pasobanda.
    Gestiona el enmarcado de tramas físicas, la inyección de errores (AWGN),
    la codificación FEC y las visualizaciones estadísticas (Ojo y BER).
    """
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
        """Construcción de panel de control lateral con scroll dinámico."""
        ctrl_outer = tk.Frame(self, bg=BG2)
        ctrl_outer.pack(side="left", fill="y", padx=(8,4), pady=8)

        ctrl_canvas = tk.Canvas(ctrl_outer, bg=BG2, highlightthickness=0, width=280)
        ctrl_canvas.pack(side="left", fill="y", expand=True)

        ctrl_sb = tk.Scrollbar(ctrl_outer, orient="vertical", command=ctrl_canvas.yview)
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

        def _on_mousewheel(e):
            ctrl_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        label(ctrl, "MÓDEM DIGITAL", fg=YELLOW, font=FONT_TITLE).pack(pady=(12,4))
        separator(ctrl).pack(fill="x", pady=4)

        info = styled_frame(ctrl); info.pack(fill="x", padx=8, pady=4)
        for txt in [f"Modulación: BFSK Constante", f"Frecuencia Mark:  {F_MARK} Hz",
                    f"Frecuencia Space: {F_SPACE} Hz", f"Symbol Rate: 300 Bd", f"Fs Física: {FS} Hz"]:
            label(info, txt, fg=FG2, font=FONT_MONO).pack(anchor="w")

        separator(ctrl).pack(fill="x", pady=4)

        f_file = styled_frame(ctrl); f_file.pack(fill="x", padx=8, pady=4)
        label(f_file, "Carga Útil (Payload):").pack(anchor="w")
        row = tk.Frame(f_file, bg=BG2); row.pack(fill="x")
        self.lbl_file = label(row, "(ninguno)", fg=FG2, font=FONT_MONO)
        self.lbl_file.pack(side="left", expand=True, anchor="w")
        btn(row, "Examinar", self._load_file, color=BG3).pack(side="right")

        f_fec = styled_frame(ctrl); f_fec.pack(fill="x", padx=8, pady=4)
        tk.Checkbutton(f_fec, text="Activar Forward Error Correction\n(Hamming 7,4)",
                       variable=self.use_fec, bg=BG2, fg=FG, selectcolor=BG3,
                       activebackground=BG2, activeforeground=FG, font=FONT_LABEL).pack(anchor="w")

        f_dur = styled_frame(ctrl); f_dur.pack(fill="x", padx=8, pady=4)
        label(f_dur, "Ventana de Captura RX [s]:").pack(anchor="w")
        self.dur_var = tk.StringVar(value="30")
        entry(f_dur, textvariable=self.dur_var).pack(fill="x")

        separator(ctrl).pack(fill="x", pady=6)
        btn(ctrl, "Transmisión Acústica (TX)", self._run_tx,  ACCENT).pack(fill="x", padx=8, pady=3)
        btn(ctrl, "Exportar Trama Física a WAV", self._save_tx_wav, BG3).pack(fill="x", padx=8, pady=3)

        separator(ctrl).pack(fill="x", pady=4)

        # ── Control de Inyección de Ruido AWGN ──
        f_snr = styled_frame(ctrl); f_snr.pack(fill="x", padx=8, pady=4)
        snr_row = tk.Frame(f_snr, bg=BG2); snr_row.pack(fill="x")
        label(snr_row, "Degradación de Canal (SNR):", fg=FG2).pack(side="left")
        self.snr_val_lbl = label(snr_row, "20 dB", fg=ACCENT, font=FONT_MONO)
        self.snr_val_lbl.pack(side="right")
        self.snr_var = tk.DoubleVar(value=20.0)
        snr_scale = tk.Scale(f_snr, from_=0, to=40, resolution=1, orient="horizontal", variable=self.snr_var,
                             bg=BG2, fg=FG2, troughcolor=BG3, highlightthickness=0, showvalue=False,
                             command=lambda v: self.snr_val_lbl.config(text=f"{int(float(v))} dB"))
        snr_scale.pack(fill="x", pady=2)
        btn(ctrl, "Inyectar AWGN y Exportar", self._save_noisy_wav, BG3).pack(fill="x", padx=8, pady=3)

        separator(ctrl).pack(fill="x", pady=4)
        btn(ctrl, "Recepción Acústica (Micrófono)", self._run_rx,  ACCENT2).pack(fill="x", padx=8, pady=3)
        btn(ctrl, "Demodular Trama Offline (Archivo)", self._run_rx_file, ACCENT2).pack(fill="x", padx=8, pady=3)
        btn(ctrl, "Análisis de Desempeño (BER/Ojo)", self._run_ber, BG3).pack(fill="x", padx=8, pady=3)

        separator(ctrl).pack(fill="x", pady=6)
        self.rx_result_lbl = label(ctrl, "", fg=FG2, font=FONT_MONO)
        self.rx_result_lbl.pack(padx=8, anchor="w")
        btn(ctrl, "Guardar Payload Recuperado", self._save_rx, BG3).pack(fill="x", padx=8, pady=3)

        label(ctrl, "Registro de Eventos:", fg=FG2).pack(anchor="w", padx=8)
        self.log = tk.Text(ctrl, height=14, width=38, bg=BG3, fg=FG2, font=FONT_MONO, relief="flat", state="disabled")
        self.log.pack(fill="x", padx=8, pady=4)

        self.plot_frame = tk.Frame(self, bg=BG)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=(4,8), pady=8)
        self.canvas = None

    def _log(self, msg: str):
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
            self._log(f"Target apuntado: {os.path.basename(p)} ({size_kb:.1f} KB)")

    def _run_tx(self):
        """Construye las cabeceras MAC y despliega la transmisión física."""
        if self.file_path.get() == "(ninguno)":
            messagebox.showwarning("Advertencia", "No existe carga útil. Seleccione un archivo.")
            return

        def task():
            self._log("Empaquetando trama y aplicando hash...")
            result = transmit(self.file_path.get(), use_fec=self.use_fec.get(), progress_cb=self._log)
            self.tx_audio = result["audio"]
            self._log(f"✓ Bloque de bytes despachado: {result['bytes_sent']} B")
            self._log(f"  Tasa efectiva: {result['baud_rate']:.0f} bps")
            
            fig = plot_digital_tx(result["audio"], FS, F_MARK, F_SPACE)
            self.after(0, lambda: self._show_figure(fig))

        threading.Thread(target=task, daemon=True).start()

    def _run_rx(self):
        try:
            dur = float(self.dur_var.get())
        except ValueError:
            return

        def task():
            self._log(f"Activando escucha asíncrona ({dur}s)...")
            result = receive(dur, progress_cb=self._log)
            self.rx_audio   = result.get("audio")
            self.rx_data    = result.get("data")
            self.rx_result  = result

            if result.get("success"):
                self.after(0, lambda: self.rx_result_lbl.config(
                    text=f"✓ Validado: {result['length']} bytes | FEC: {result['corrections']} bits", fg=ACCENT))
                self._log("✓ Integridad SHA-256 confirmada.")
            else:
                self.after(0, lambda: self.rx_result_lbl.config(text="✗ Corrupción de trama en capa MAC", fg=RED))
                self._log("✗ Descarte de trama preventivo.")

        threading.Thread(target=task, daemon=True).start()

    def _run_ber(self):
        """Simula canal discreto para trazar métricas de confiabilidad estadística."""
        def task():
            self._log("Lanzando simulación Monte Carlo para curvas BER...")
            snr = np.linspace(-2, 14, 30)
            ber_data = compute_ber_curve(snr)
            audio = self.tx_audio if self.tx_audio is not None else np.sin(2*np.pi*F_MARK*np.arange(FS)//FS)

            fig = plot_ber_and_eye(ber_data, audio, FS, SPB)
            self._log("✓ Renderizado analítico completado.")
            self.after(0, lambda: self._show_figure(fig))

        threading.Thread(target=task, daemon=True).start()

    def _save_rx(self):
        if self.rx_data is None: return
        path = filedialog.asksaveasfilename()
        if path:
            with open(path, 'wb') as f:
                f.write(self.rx_data)
            self._log(f"✓ Extracción exitosa al sistema operativo: {os.path.basename(path)}")

    def _save_tx_wav(self):
        if self.tx_audio is None: return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV PCM", "*.wav")])
        if path:
            save_modulated_wav(path, self.tx_audio, FS)
            self._log(f"✓ Bloque físico volcado en disco: {os.path.basename(path)}")

    def _save_noisy_wav(self):
        """Permite generar archivos de prueba artificialmente degradados."""
        if self.tx_audio is None: return
        snr_db = self.snr_var.get()
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV PCM", "*.wav")],
                                            title=f"Guardar archivo contaminado AWGN ({int(snr_db)} dB)")
        if not path: return
        noisy = add_awgn(self.tx_audio, snr_db)
        save_modulated_wav(path, noisy, FS)
        
        signal_power = np.mean(self.tx_audio ** 2)
        noise_power  = np.mean((noisy - self.tx_audio) ** 2)
        snr_real     = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        self._log(f"✓ Volcado AWGN: {os.path.basename(path)} | Margen de Ruido Real: {snr_real:.1f} dB")

    def _run_rx_file(self):
        """Bypass del hardware de micrófono para forzar la lectura de un bloque offline."""
        path = filedialog.askopenfilename(filetypes=[("WAV PCM", "*.wav")])
        if not path: return

        def task():
            try:
                self._log(f"Lanzando decodificación estática: {os.path.basename(path)}")
                result = receive_from_file(path, progress_cb=self._log)
                self.rx_audio  = result.get("audio")
                self.rx_data   = result.get("data")
                self.rx_result = result

                if result.get("success"):
                    self.after(0, lambda: self.rx_result_lbl.config(
                        text=f"✓ Validado: {result['length']} bytes | FEC: {result['corrections']} bits", fg=ACCENT))
                    self._log(f"✓ Extracción de capa física completada.")
                    fig = plot_digital_tx(self.rx_audio, result["fs"], F_MARK, F_SPACE)
                    self.after(0, lambda: self._show_figure(fig))
                else:
                    self.after(0, lambda: self.rx_result_lbl.config(text="✗ Excepción CRC o Checksum", fg=RED))
                    self._log(f"✗ Falla en lectura: {result.get('error', 'Causa desconocida')}")
            except Exception as e:
                self._log(f"✗ Desborde en RX: {str(e)}")

        threading.Thread(target=task, daemon=True).start()

# =============================================================================
# Clase Raíz de la Aplicación
# =============================================================================

class App(tk.Tk):
    """Contenedor maestro del simulador de comunicaciones."""
    def __init__(self):
        super().__init__()
        self.title("EL5522 — Entorno de Simulación DSP")
        self.configure(bg=BG)
        self.geometry("1280x760")
        self.resizable(True, True)
        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG3, foreground=FG2, padding=(14, 6), font=FONT_BUTTON)
        style.map("TNotebook.Tab", background=[("selected", BG2)], foreground=[("selected", FG)])
        style.configure("Dark.TCombobox", fieldbackground=BG3, background=BG3, foreground=FG, arrowcolor=FG,
                        selectbackground=BG3, selectforeground=FG)

    def _build_ui(self):
        header = tk.Frame(self, bg=BG, pady=8)
        header.pack(fill="x", padx=12)
        tk.Label(header, text="TALLER DE COMUNICACIONES ELÉCTRICAS — EL5522",
                 bg=BG, fg=ACCENT2, font=("Consolas", 14, "bold")).pack(side="left")
        tk.Label(header, text="Arquitectura de Enrutamiento Analógico y Digital",
                 bg=BG, fg=FG2, font=FONT_LABEL).pack(side="left", padx=(14,0))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_ssb  = SSBTab(nb)
        self.tab_dig  = DigitalTab(nb)

        nb.add(self.tab_ssb, text="  Transceptor Analógico SSB/ISB  ")
        nb.add(self.tab_dig, text="  Módem Digital Pasobanda BFSK  ")

        footer = tk.Frame(self, bg=BG3, pady=3)
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, text="Instituto Tecnológico de Costa Rica · Escuela de Ingeniería Electrónica · 2026",
                 bg=BG3, fg=FG2, font=FONT_MONO).pack()

# =============================================================================
# Lanzador de la Aplicación
# =============================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
