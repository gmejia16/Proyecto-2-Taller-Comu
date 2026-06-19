"""
Módulo de Modulación y Demodulación SSB/ISB.

Este módulo implementa el método de discriminación de fase basado en la 
Transformada de Hilbert para la generación de señales de Banda Lateral Única (SSB) 
y Banda Lateral Independiente (ISB). Incluye capacidades de demodulación síncrona
con simulación de canal (errores de fase y frecuencia) e interfaces de 
hardware acústico.

Curso: EL5522 - Taller de Comunicaciones Eléctricas
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import scipy.io.wavfile as wav
import sounddevice as sd

# =============================================================================
# Utilidades de Procesamiento de Audio
# =============================================================================

def load_wav(filepath: str) -> tuple[np.ndarray, int]:
    """
    Carga un archivo de audio WAV y lo acondiciona para el procesamiento matemático.
    Convierte señales estéreo a mono y normaliza la amplitud al rango [-1.0, 1.0].

    Parameters
    ----------
    filepath : str
        Ruta absoluta o relativa del archivo .wav a cargar.

    Returns
    -------
    tuple[np.ndarray, int]
        - Arreglo unidimensional de tipo float64 con la señal normalizada.
        - Frecuencia de muestreo (fs) del archivo original.
    """
    fs, data = wav.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]  # Extracción exclusiva del canal izquierdo (mono)
        
    data = data.astype(np.float64)
    # Normalización con término epsilon (1e-12) para evitar división por cero en silencios puros
    data /= np.max(np.abs(data)) + 1e-12
    
    return data, fs

def save_wav(filepath: str, signal: np.ndarray, fs: int) -> None:
    """
    Exporta una señal matemática a un archivo de audio WAV PCM estándar de 16 bits.

    Parameters
    ----------
    filepath : str
        Ruta destino donde se guardará el archivo.
    signal : np.ndarray
        Arreglo con la señal a exportar.
    fs : int
        Frecuencia de muestreo a registrar en la cabecera del WAV.
    """
    signal = signal / (np.max(np.abs(signal)) + 1e-12)
    signal_int = (signal * 32767).astype(np.int16)
    wav.write(filepath, fs, signal_int)

def lowpass_filter(signal: np.ndarray, cutoff: float, fs: int, order: int = 6) -> np.ndarray:
    """
    Aplica un filtro digital pasa-bajos tipo Butterworth con filtrado bidireccional (fase nula).

    Parameters
    ----------
    signal : np.ndarray
        Señal de entrada a filtrar.
    cutoff : float
        Frecuencia de corte a -3dB en Hz.
    fs : int
        Frecuencia de muestreo del sistema en Hz.
    order : int, opcional
        Orden del filtro Butterworth. Por defecto es 6.

    Returns
    -------
    np.ndarray
        Señal filtrada sin distorsión de fase.
    """
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='low')
    # filtfilt aplica el filtro hacia adelante y hacia atrás, garantizando retardo de grupo cero
    return filtfilt(b, a, signal)

# =============================================================================
# Núcleo de Modulación Analógica
# =============================================================================

def modulate_ssb(message: np.ndarray, fs: int, fc: float,
                 mode: str = "SSB-SC", sideband: str = "USB",
                 Ac: float = 1.0) -> dict:
    """
    Modula una señal en Banda Lateral Única utilizando la Transformada de Hilbert.

    Parameters
    ----------
    message : np.ndarray
        Señal de mensaje (banda base) normalizada.
    fs : int
        Frecuencia de muestreo del vector.
    fc : float
        Frecuencia de la portadora en Hz.
    mode : str, opcional
        Tipo de portadora: "SSB-SC" (suprimida) o "SSB-FC" (completa). 
        Por defecto "SSB-SC".
    sideband : str, opcional
        Banda a transmitir: "USB" (superior) o "LSB" (inferior). Por defecto "USB".
    Ac : float, opcional
        Amplitud de la portadora inyectada (solo para modo FC). Por defecto 1.0.

    Returns
    -------
    dict
        Diccionario analítico con las señales resultantes y metadatos del sistema.
    """
    t = np.arange(len(message)) / fs
    carrier = np.cos(2 * np.pi * fc * t)

    # 1. Generación de la Señal Analítica
    # scipy.signal.hilbert retorna: m(t) + j*m_hat(t)
    analytic  = hilbert(message)
    msg_hat   = np.imag(analytic)    # m_hat(t): Señal desfasada exactamente 90 grados

    # 2. Modulación en Doble Banda Lateral (DSB-SC)
    dsb_sc = message * carrier

    # 3. Ecuación General del Método de Discriminación de Fase (SSB)
    # s_ssb(t) = m(t)*cos(wc*t) -/+ m_hat(t)*sin(wc*t)
    if sideband == "USB":
        # Resta cuadratura para cancelar la banda inferior
        ssb_sc = 0.5 * (message * np.cos(2 * np.pi * fc * t) - msg_hat * np.sin(2 * np.pi * fc * t))
    else:  # LSB
        # Suma cuadratura para cancelar la banda superior
        ssb_sc = 0.5 * (message * np.cos(2 * np.pi * fc * t) + msg_hat * np.sin(2 * np.pi * fc * t))

    # 4. Inyección de Portadora (Si es requerido por el modo)
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
    Modulación de Banda Lateral Independiente (ISB).
    Multiplexa dos mensajes de audio distintos en la misma portadora frecuencial,
    ubicando uno en la USB y otro en la LSB.

    Parameters
    ----------
    msg_upper : np.ndarray
        Mensaje a transmitir en la Banda Lateral Superior.
    msg_lower : np.ndarray
        Mensaje a transmitir en la Banda Lateral Inferior.
    fs : int
        Frecuencia de muestreo.
    fc : float
        Frecuencia de la portadora central.
    Ac : float, opcional
        Amplitud del tono piloto/portadora inyectada. Por defecto 1.0.

    Returns
    -------
    dict
        Diccionario con las componentes individuales y la señal ISB compuesta.
    """
    # Truncamiento de vectores a la longitud del mensaje más corto
    n = min(len(msg_upper), len(msg_lower))
    msg_upper, msg_lower = msg_upper[:n], msg_lower[:n]
    t = np.arange(n) / fs

    # Generación ortogonal de bandas independientes
    usb = modulate_ssb(msg_upper, fs, fc, mode="SSB-SC", sideband="USB")["ssb"]
    lsb = modulate_ssb(msg_lower, fs, fc, mode="SSB-SC", sideband="LSB")["ssb"]
    
    # Composición de la señal final con piloto de sincronización
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

# =============================================================================
# Núcleo de Demodulación Analógica
# =============================================================================

def demodulate_ssb(ssb: np.ndarray, fs: int, fc: float,
                   phase_error_deg: float = 0.0,
                   freq_error_hz: float = 0.0,
                   sideband: str = "USB") -> dict:
    """
    Demodulador síncrono coherente para señales SSB.
    Simula las no idealidades del canal y del oscilador local del receptor.

    Parameters
    ----------
    ssb : np.ndarray
        Señal SSB recibida del canal.
    fs : int
        Frecuencia de muestreo.
    fc : float
        Frecuencia nominal de la portadora original.
    phase_error_deg : float, opcional
        Desfase del oscilador local en grados. Por defecto 0.0.
    freq_error_hz : float, opcional
        Desviación de frecuencia del oscilador local en Hz. Por defecto 0.0.
    sideband : str, opcional
        Banda esperada en recepción. Por defecto "USB".

    Returns
    -------
    dict
        Diccionario con el mensaje de audio recuperado y parámetros de error.
    """
    t = np.arange(len(ssb)) / fs
    phi  = np.deg2rad(phase_error_deg)
    fc_r = fc + freq_error_hz  # Frecuencia real de la portadora local

    # 1. Multiplicación Síncrona (Mezclado)
    local_carrier = np.cos(2 * np.pi * fc_r * t + phi)
    mixed = ssb * local_carrier

    # 2. Extracción de Banda Base (Filtrado)
    # Tras el mezclado, el espectro se desdobla en banda base y en 2*fc.
    # El filtro elimina la componente de alta frecuencia (2*fc).
    bw = fc * 0.6  # Ancho de banda de corte diseñado de forma conservadora
    recovered = lowpass_filter(mixed, bw, fs)
    
    # 3. Compensación de Potencia
    # La multiplicación coherente divide la amplitud teórica a la mitad
    recovered *= 2.0       

    # Normalización final de salida
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
    """
    Demodulador de Banda Lateral Independiente.
    Extrae simultáneamente los dos canales de información de la señal recibida.
    """
    usb_result = demodulate_ssb(isb, fs, fc, phase_error_deg, freq_error_hz, "USB")
    lsb_result = demodulate_ssb(isb, fs, fc, phase_error_deg, freq_error_hz, "LSB")

    return {
        "recovered_upper": usb_result["recovered"],
        "recovered_lower": lsb_result["recovered"],
        "t": usb_result["t"],
        "fs": fs,
    }

# =============================================================================
# Interfaces de Transceptor Acústico Físico
# =============================================================================

def transmit_ssb(mod_result: dict, progress_cb=None) -> float:
    """
    Inyecta la señal SSB modulada directamente al hardware de salida (parlante).
    Válido operativamente siempre que fc + Ancho de Banda del mensaje < 22 kHz.
    
    Parameters
    ----------
    mod_result : dict
        Diccionario generado por las funciones de modulación.
    progress_cb : callable, opcional
        Función de callback para registro de logs en GUI.

    Returns
    -------
    float
        Duración total de la transmisión acústica en segundos.
    """
    ssb = mod_result["ssb"]
    fs  = mod_result["fs"]
    fc  = mod_result["fc"]

    ssb_norm = ssb / (np.max(np.abs(ssb)) + 1e-12)
    duration = len(ssb_norm) / fs

    if progress_cb:
        progress_cb(f"Transmitiendo SSB por parlante — fc={fc} Hz, {duration:.1f}s...")

    sd.play(ssb_norm.astype(np.float32), samplerate=fs)
    sd.wait()

    if progress_cb:
        progress_cb("✓ Transmisión SSB terminada.")

    return duration

def receive_ssb(fs: int, fc: float, duration_s: float,
                sideband: str = "USB",
                phase_error_deg: float = 0.0,
                freq_error_hz: float = 0.0,
                progress_cb=None) -> dict:
    """
    Captura audio crudo desde el hardware de entrada (micrófono) y lo demodula en tiempo real.

    Parameters
    ----------
    fs : int
        Frecuencia de muestreo de captura.
    fc : float
        Frecuencia de portadora configurada en el receptor.
    duration_s : float
        Tiempo de escucha continua del canal acústico.
    sideband, phase_error_deg, freq_error_hz : 
        Parámetros del demodulador coherente.

    Returns
    -------
    dict
        Diccionario con la señal recibida del canal y el mensaje banda base recuperado.
    """
    if progress_cb:
        progress_cb(f"Grabando {duration_s}s por micrófono (fs={fs} Hz)...")

    n_samples = int(duration_s * fs)
    recorded  = sd.rec(n_samples, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    ssb_rx = recorded.flatten().astype(np.float64)

    peak = np.max(np.abs(ssb_rx))
    if peak > 1e-6:
        ssb_rx /= peak

    if progress_cb:
        progress_cb("Demodulando señal acústica recibida...")

    demod = demodulate_ssb(ssb_rx, fs, fc,
                           phase_error_deg=phase_error_deg,
                           freq_error_hz=freq_error_hz,
                           sideband=sideband)
    demod["ssb_rx"] = ssb_rx
    return demod
