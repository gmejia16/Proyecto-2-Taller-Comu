import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# BLOQUE 1: LÓGICA MATEMÁTICA Y PROCESAMIENTO

def procesar_y_modular(ruta_wav, fc, tipo_banda, fs_objetivo=96000):
    """
    Carga el audio, aplica sobremuestreo (resampling) para evitar aliasing,
    y modula la señal en SSB (USB o LSB).
    """
    # 1. Carga del archivo original
    fs_original, data = wavfile.read(ruta_wav)
    
    # Convertir a mono si es estéreo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Normalización de amplitud
    data = data / np.max(np.abs(data))
    
    # 2. Etapa de Resampling (Sobremuestreo)
    if fs_original < fs_objetivo:
        data_resampled = resample_poly(data, fs_objetivo, fs_original)
        fs_actual = fs_objetivo
    else:
        data_resampled = data
        fs_actual = fs_original

    # 3. Núcleo del Modulador SSB (Transformada de Hilbert)
    t = np.arange(len(data_resampled)) / fs_actual
    
    # Señal analítica m(t) + j*hat{m}(t)
    senal_analitica = hilbert(data_resampled)
    m_hat = np.imag(senal_analitica)

    # Portadoras ortogonales
    portadora_cos = np.cos(2 * np.pi * fc * t)
    portadora_sin = np.sin(2 * np.pi * fc * t)
    
    # Ecuación de modulación DSB
    senal_dsb = data_resampled * portadora_cos

    # Ecuación de modulación en fase y cuadratura (SSB)
    if tipo_banda == 'USB':
        senal_ssb = senal_dsb - (m_hat * portadora_sin)
    else: # LSB
        senal_ssb = senal_dsb + (m_hat * portadora_sin)
        
    return t, data_resampled, senal_dsb, senal_ssb, fs_actual


def graficar_resultados(t, mensaje, senal_dsb, senal_ssb, fs, fc):
    """
    Genera una figura con las gráficas en el dominio del tiempo y la frecuencia
    para el mensaje original, la señal DSB y la señal SSB.
    """
    # 1. Preparar el cálculo de la FFT (Dominio de la Frecuencia)
    N = len(mensaje)
    frecuencias = fftshift(fftfreq(N, 1/fs))
    
    espectro_msg = fftshift(np.abs(fft(mensaje)) / N)
    espectro_dsb = fftshift(np.abs(fft(senal_dsb)) / N)
    espectro_ssb = fftshift(np.abs(fft(senal_ssb)) / N)

    # 2. Definir una ventana de tiempo (10 ms) para visualizar la portadora
    muestras_10ms = int(fs * 0.01)
    limite_muestras = min(muestras_10ms, N)
    t_ventana = t[:limite_muestras] * 1000 # a milisegundos

    # 3. Configuración de la Figura
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.canvas.manager.set_window_title('Análisis Espectral y Temporal - Avance 1')
    fig.suptitle(f'Modulación de Amplitud (Portadora: {fc/1000:.1f} kHz)', fontsize=16)

    # --- COLUMNA 1: Dominio del Tiempo ---
    axs[0, 0].plot(t_ventana, mensaje[:limite_muestras], color='blue')
    axs[0, 0].set_title('Mensaje Original (Señal de Audio)')
    axs[0, 0].set_ylabel('Amplitud')
    axs[0, 0].set_xlabel('Tiempo (ms)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(t_ventana, senal_dsb[:limite_muestras], color='orange')
    axs[1, 0].set_title('Señal Modulada DSB-SC')
    axs[1, 0].set_ylabel('Amplitud')
    axs[1, 0].set_xlabel('Tiempo (ms)')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    axs[2, 0].plot(t_ventana, senal_ssb[:limite_muestras], color='green')
    axs[2, 0].set_title('Señal Modulada SSB')
    axs[2, 0].set_ylabel('Amplitud')
    axs[2, 0].set_xlabel('Tiempo (ms)')
    axs[2, 0].grid(True, linestyle='--', alpha=0.7)

    # --- COLUMNA 2: Dominio de la Frecuencia ---
    limite_frec = min(fc * 1.5, fs / 2)

    axs[0, 1].plot(frecuencias, espectro_msg, color='blue')
    axs[0, 1].set_title('Espectro del Mensaje')
    axs[0, 1].set_ylabel('Magnitud')
    axs[0, 1].set_xlabel('Frecuencia (Hz)')
    axs[0, 1].set_xlim(-limite_frec, limite_frec)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 1].plot(frecuencias, espectro_dsb, color='orange')
    axs[1, 1].set_title('Espectro DSB-SC (Doble Banda)')
    axs[1, 1].set_ylabel('Magnitud')
    axs[1, 1].set_xlabel('Frecuencia (Hz)')
    axs[1, 1].set_xlim(-limite_frec, limite_frec)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    axs[2, 1].plot(frecuencias, espectro_ssb, color='green')
    axs[2, 1].set_title('Espectro SSB (Banda Única)')
    axs[2, 1].set_ylabel('Magnitud')
    axs[2, 1].set_xlabel('Frecuencia (Hz)')
    axs[2, 1].set_xlim(-limite_frec, limite_frec)
    axs[2, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def exportar_audio(ruta_original, senal, fs, sufijo="_modulado"):
    """
    Normaliza y guarda un arreglo numpy como archivo de audio WAV de 16 bits.
    """
    # Generar el nombre del archivo de salida en la misma carpeta que el original
    nombre_base, _ = os.path.splitext(ruta_original)
    ruta_salida = f"{nombre_base}{sufijo}.wav"
    
    # Prevenir división por cero si la señal está vacía o es puro silencio
    max_val = np.max(np.abs(senal))
    if max_val == 0:
        max_val = 1.0
        
    # Normalizar al rango estricto de 16-bit PCM (-32768 a 32767)
    senal_normalizada = np.int16((senal / max_val) * 32767)
    
    # Escribir el archivo en el disco
    wavfile.write(ruta_salida, fs, senal_normalizada)
    
    return ruta_salida

# BLOQUE 2: INTERFAZ GRÁFICA (GUI)

class AppModuladorSSB:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador SSB - Avance 1")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        self.ruta_archivo = tk.StringVar()
        style = ttk.Style()
        style.theme_use('clam')
        
        frame = ttk.Frame(root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Archivo de Audio (.wav):", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_ruta = ttk.Entry(frame, textvariable=self.ruta_archivo, width=35, state='readonly')
        self.entry_ruta.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        btn_buscar = ttk.Button(frame, text="Buscar...", command=self.seleccionar_archivo)
        btn_buscar.grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(frame, text="Frecuencia Portadora (Hz):", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=10)
        self.entry_fc = ttk.Entry(frame, width=15)
        self.entry_fc.insert(0, "20000")
        self.entry_fc.grid(row=2, column=1, sticky=tk.W, pady=10)
        
        ttk.Label(frame, text="Banda Lateral:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.combo_banda = ttk.Combobox(frame, values=["USB", "LSB"], width=12, state="readonly")
        self.combo_banda.set("USB")
        self.combo_banda.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        self.btn_modular = ttk.Button(frame, text="Ejecutar Modulación", command=self.ejecutar_procesamiento)
        self.btn_modular.grid(row=4, column=0, columnspan=3, pady=25)
        
    def seleccionar_archivo(self):
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=[("Archivos WAV", "*.wav")]
        )
        if archivo:
            self.ruta_archivo.set(archivo)
            
    def ejecutar_procesamiento(self):
        if not self.ruta_archivo.get():
            messagebox.showerror("Error", "Por favor, seleccione un archivo de audio .wav primero.")
            return
            
        try:
            fc = float(self.entry_fc.get())
            if fc <= 0 or fc > 25000:
                raise ValueError("Fuera de rango estipulado")
        except ValueError:
            messagebox.showerror("Error", "La frecuencia de portadora debe ser un número entre 0 y 25000 Hz (Máx 25 kHz).")
            return
            
        tipo_banda = self.combo_banda.get()
        
        # Bloque TRY unificado y corregido
        try:
            # 1. Ejecución del pipeline matemático
            t, mensaje, senal_dsb, senal_ssb, fs_final = procesar_y_modular(
                self.ruta_archivo.get(), fc, tipo_banda
            )
            
            # 2. Exportar el audio modulado al disco
            ruta_guardada = exportar_audio(self.ruta_archivo.get(), senal_ssb, fs_final, f"_modulado_{int(fc)}Hz_{tipo_banda}")
            
            # 3. Confirmación de éxito actualizada
            info_msg = (
                f"Modulación completada exitosamente.\n\n"
                f"Frecuencia de muestreo final: {fs_final} Hz\n"
                f"Modo: {tipo_banda}\n\n"
                f"Audio guardado exitosamente en:\n{ruta_guardada}\n\n"
                f"Cerrando este mensaje se abrirán las gráficas."
            )
            messagebox.showinfo("Procesamiento Completado", info_msg)
            
            # 4. Lanzar la ventana de Matplotlib
            graficar_resultados(t, mensaje, senal_dsb, senal_ssb, fs_final, fc)
            
        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Ocurrió un error:\n{str(e)}")

# BLOQUE 3: PUNTO DE ENTRADA MAIN

if __name__ == "__main__":
    root = tk.Tk()
    app = AppModuladorSSB(root)
    root.mainloop()
