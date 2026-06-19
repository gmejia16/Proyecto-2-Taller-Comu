"""
Módulo de Modulación y Demodulación Digital Pasobanda (BFSK).

Este módulo implementa la capa física y la capa de enlace de datos para un
sistema de transmisión acústica. Incluye codificación de canal (Hamming 7,4),
generación de tramas con sincronización y validación (SHA-256), y modulación
por desplazamiento de frecuencia binaria (BFSK) no coherente.

Curso: EL5522 - Taller de Comunicaciones Eléctricas
"""

import numpy as np
from scipy.signal import butter, filtfilt
import sounddevice as sd
import struct
import hashlib
import time

# =============================================================================
# Parámetros Globales del Sistema
# =============================================================================

FS        = 44100       # Frecuencia de muestreo del audio [Hz]
F_MARK    = 1200        # Frecuencia para el bit lógico 1 (Mark) [Hz]
F_SPACE   = 2200        # Frecuencia para el bit lógico 0 (Space) [Hz]
BAUD_RATE = 300         # Tasa de transmisión de símbolos [Baudios]
SPB       = FS // BAUD_RATE       # Muestras por bit (Samples Per Bit)
PREAMBLE  = b'\xAA\xAA\xAA\xAA'   # Secuencia alternante para ajuste de ganancia
SYNC_WORD = b'\x16\x16\xF0\xF0'   # Palabra única para alineación de trama

# =============================================================================
# Codificación de Canal (Forward Error Correction - FEC)
# =============================================================================

# Matriz generadora G para código Hamming(7,4) en forma sistemática
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

# Matriz de comprobación de paridad H
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)

def hamming_encode(nibble: np.ndarray) -> np.ndarray:
    """
    Codifica un bloque de 4 bits utilizando código Hamming(7,4).

    Parameters
    ----------
    nibble : np.ndarray
        Arreglo de 4 elementos (bits) correspondientes a los datos originales.

    Returns
    -------
    np.ndarray
        Arreglo de 7 elementos codificados (4 datos + 3 paridad).
    """
    return (nibble @ G) % 2

def hamming_decode(codeword: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Decodifica un bloque de 7 bits y corrige errores simples si existen.

    Parameters
    ----------
    codeword : np.ndarray
        Arreglo de 7 bits recibido del canal.

    Returns
    -------
    tuple[np.ndarray, bool]
        - Arreglo de 4 bits decodificados.
        - Booleano indicando si se detectó y corrigió un error.
    """
    syndrome = (H @ codeword) % 2
    # El valor decimal del síndrome indica la posición del error (1 a 7)
    error_pos = syndrome[0]*4 + syndrome[1]*2 + syndrome[2]*1
    
    corrected = False
    if error_pos != 0:
        codeword = codeword.copy()
        codeword[error_pos - 1] ^= 1  # Inversión del bit erróneo
        corrected = True
        
    return codeword[:4], corrected

def encode_bytes(data: bytes) -> np.ndarray:
    """
    Aplica codificación Hamming(7,4) a una secuencia completa de bytes.

    Parameters
    ----------
    data : bytes
        Información en bruto a transmitir.

    Returns
    -------
    np.ndarray
        Arreglo unidimensional de bits codificados.
    """
    bits_out = []
    for byte in data:
        # Separación del byte en dos bloques de 4 bits (nibbles)
        nibble_hi = np.array([(byte >> (7 - i)) & 1 for i in range(4)], dtype=np.uint8)
        nibble_lo = np.array([(byte >> (3 - i)) & 1 for i in range(4)], dtype=np.uint8)
        
        bits_out.extend(hamming_encode(nibble_hi).tolist())
        bits_out.extend(hamming_encode(nibble_lo).tolist())
        
    return np.array(bits_out, dtype=np.uint8)

def decode_bits(bits: np.ndarray) -> tuple[bytes, int]:
    """
    Decodifica una secuencia de bits Hamming(7,4) y la reensambla en bytes.

    Parameters
    ----------
    bits : np.ndarray
        Arreglo de bits recibidos.

    Returns
    -------
    tuple[bytes, int]
        - Datos reconstruidos en formato bytes.
        - Contador total de bits corregidos durante la decodificación.
    """
    result = []
    corrections = 0
    
    for i in range(0, len(bits), 14):
        if i + 14 > len(bits):
            break
            
        hi_code = bits[i:i+7]
        lo_code = bits[i+7:i+14]
        
        hi, c1 = hamming_decode(hi_code)
        lo, c2 = hamming_decode(lo_code)
        
        if c1: corrections += 1
        if c2: corrections += 1
        
        # Reconstrucción del byte a partir de los dos nibbles corregidos
        byte_val = 0
        for b in hi: byte_val = (byte_val << 1) | int(b)
        for b in lo: byte_val = (byte_val << 1) | int(b)
        
        result.append(byte_val)
        
    return bytes(result), corrections

# =============================================================================
# Capa Física: Modulación y Demodulación BFSK
# =============================================================================

def bits_to_audio(bits: np.ndarray, fs: int = FS,
                  f1: float = F_MARK, f0: float = F_SPACE,
                  spb: int = SPB) -> np.ndarray:
    """
    Sintetiza la señal acústica BFSK a partir de un flujo de bits.

    Parameters
    ----------
    bits : np.ndarray
        Secuencia lógica a modular.
    fs : int, opcional
        Frecuencia de muestreo. Por defecto FS.
    f1 : float, opcional
        Frecuencia del tono Mark. Por defecto F_MARK.
    f0 : float, opcional
        Frecuencia del tono Space. Por defecto F_SPACE.
    spb : int, opcional
        Número de muestras por símbolo. Por defecto SPB.

    Returns
    -------
    np.ndarray
        Señal analógica modulada en formato de arreglo numpy.
    """
    t_bit = np.arange(spb) / fs
    audio = []
    
    # Enventanado Hanning para suavizar transiciones y reducir el ensanchamiento 
    # espectral (spectral leakage) ocasionado por discontinuidades de fase.
    window = np.hanning(spb)
    
    for bit in bits:
        freq = f1 if bit == 1 else f0
        tone = np.sin(2 * np.pi * freq * t_bit)
        audio.append(tone * np.sqrt(window))
        
    return np.concatenate(audio)

def build_frame(data: bytes, use_fec: bool = True) -> np.ndarray:
    """
    Construye la estructura completa de la trama MAC y genera el audio.
    
    Estructura de Trama:
    [PREAMBLE (4B)] | [SYNC (4B)] | [LENGTH (4B)] | [FEC_FLAG (1B)] | [SHA256 (4B)] | [PAYLOAD]

    Parameters
    ----------
    data : bytes
        Carga útil (payload) a transmitir.
    use_fec : bool, opcional
        Indicador para activar codificación Hamming. Por defecto True.

    Returns
    -------
    np.ndarray
        Señal de audio lista para ser inyectada al hardware.
    """
    checksum = hashlib.sha256(data).digest()[:4]
    fec_flag = b'\x01' if use_fec else b'\x00'
    length   = struct.pack('>I', len(data))

    if use_fec:
        payload = encode_bytes(data)
    else:
        payload = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    header_bytes = PREAMBLE + SYNC_WORD + length + fec_flag + checksum
    header_bits  = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))

    all_bits = np.concatenate([header_bits, payload])
    return bits_to_audio(all_bits)

def bandpass(signal: np.ndarray, fc: float, bw: float, fs: int) -> np.ndarray:
    """
    Aplica un filtro pasa-banda digital tipo Butterworth de orden 4.

    Parameters
    ----------
    signal : np.ndarray
        Señal de entrada a filtrar.
    fc : float
        Frecuencia central del filtro.
    bw : float
        Ancho de banda del filtro.
    fs : int
        Frecuencia de muestreo.

    Returns
    -------
    np.ndarray
        Señal filtrada.
    """
    nyq = fs / 2
    low  = max((fc - bw/2) / nyq, 0.001)
    high = min((fc + bw/2) / nyq, 0.999)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def audio_to_bits(audio: np.ndarray, fs: int = FS,
                  f1: float = F_MARK, f0: float = F_SPACE,
                  spb: int = SPB) -> np.ndarray:
    """
    Demodulador no coherente basado en detección de energía.

    Aplica filtros pasa-banda independientes para las frecuencias Mark y Space,
    comparando la energía espectral en cada intervalo de símbolo.

    Parameters
    ----------
    audio : np.ndarray
        Señal acústica recibida.
    fs : int, opcional
        Frecuencia de muestreo.
    f1, f0 : float, opcional
        Frecuencias de los tonos correspondientes.
    spb : int, opcional
        Muestras por símbolo.

    Returns
    -------
    np.ndarray
        Secuencia de bits demodulada.
    """
    # Ancho de banda de los filtros ajustado al 80% de la separación tonal
    bw = abs(f1 - f0) * 0.8          
    sig_mark  = bandpass(audio, f1, bw, fs)
    sig_space = bandpass(audio, f0, bw, fs)

    bits = []
    n_bits = len(audio) // spb
    
    for i in range(n_bits):
        s = i * spb
        e_mark  = np.sum(sig_mark[s:s+spb] ** 2)
        e_space = np.sum(sig_space[s:s+spb] ** 2)
        
        # Criterio de decisión por máxima energía
        bits.append(1 if e_mark > e_space else 0)
        
    return np.array(bits, dtype=np.uint8)

def sync_bits(bits: np.ndarray) -> int:
    """
    Busca el inicio de la trama mediante correlación cruzada con la palabra 
    de sincronización (Sync Word).

    Parameters
    ----------
    bits : np.ndarray
        Flujo de bits demodulado en bruto.

    Returns
    -------
    int
        Índice del primer bit correspondiente a los datos de la trama (Header).
    """
    sync_bits_ref = np.unpackbits(np.frombuffer(SYNC_WORD, dtype=np.uint8))
    preamble_bits = np.unpackbits(np.frombuffer(PREAMBLE, dtype=np.uint8))
    search = np.concatenate([preamble_bits, sync_bits_ref])
    n = len(search)

    best_score = -1
    best_idx   = 0

    # Limitación de búsqueda temporal para optimización de recursos (evitar O(n²))
    search_limit = min(len(bits) - n, 2000)
    if search_limit <= 0:
        return n

    for i in range(search_limit):
        score = np.sum(bits[i:i+n] == search)
        if score > best_score:
            best_score = score
            best_idx   = i

    return best_idx + n

def decode_frame(bits: np.ndarray) -> dict:
    """
    Procesa el flujo de bits alineado, extrae las cabeceras y decodifica el payload.

    Parameters
    ----------
    bits : np.ndarray
        Flujo de bits a analizar.

    Returns
    -------
    dict
        Diccionario de estado de la recepción (éxito, datos, validación).
    """
    start = sync_bits(bits)
    header_bits = bits[start:start + 9*8]

    if len(header_bits) < 72:
        return {"success": False, "error": "Trama incompleta en recepción"}

    header_bytes = np.packbits(header_bits).tobytes()
    length   = struct.unpack('>I', header_bytes[0:4])[0]
    use_fec  = header_bytes[4] == 0x01
    checksum = header_bytes[5:9]

    # Prevención de desbordamiento de memoria
    if length == 0 or length > 1_000_000:
        return {"success": False, "error": f"Longitud de trama inválida: {length}"}

    payload_start  = start + 9*8
    n_payload_bits = length * 14 if use_fec else length * 8
    payload_bits   = bits[payload_start:payload_start + n_payload_bits]

    if len(payload_bits) < n_payload_bits:
        return {"success": False,
                "error": f"Payload truncado: {len(payload_bits)}/{n_payload_bits} bits."}

    if use_fec:
        data, corrections = decode_bits(payload_bits)
    else:
        data = np.packbits(payload_bits).tobytes()[:length]
        corrections = 0

    # Verificación de integridad por hash criptográfico
    expected_checksum = hashlib.sha256(data).digest()[:4]
    ok = (expected_checksum == checksum)

    return {
        "success":      ok,
        "data":         data if ok else None,
        "length":       length,
        "use_fec":      use_fec,
        "corrections":  corrections,
        "checksum_ok":  ok,
        "error":        None if ok else "Fallo en validación Checksum. Trama corrupta.",
    }

# =============================================================================
# Simulador de Canal y Utilidades de Transceptor
# =============================================================================

def add_awgn(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Inyecta Ruido Blanco Gaussiano Aditivo (AWGN) a la señal temporal.

    Parameters
    ----------
    audio : np.ndarray
        Señal original.
    snr_db : float
        Relación Señal a Ruido objetivo expresada en decibelios.

    Returns
    -------
    np.ndarray
        Señal resultante contaminada con ruido.
    """
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio.copy()
        
    snr_linear  = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise       = np.sqrt(noise_power) * np.random.randn(len(audio))
    
    # Recorte para evitar superación de límites dinámicos del audio hardware
    return np.clip(audio + noise, -1.0, 1.0)

def save_modulated_wav(filepath: str, audio: np.ndarray, fs: int = FS) -> None:
    """Exporta el arreglo matemático de la señal a un archivo de audio WAV PCM de 16 bits."""
    import wave
    audio_norm = np.clip(audio, -1.0, 1.0)
    samples_int16 = (audio_norm * 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           
        wf.setframerate(fs)
        wf.writeframes(samples_int16.tobytes())

def load_modulated_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Carga un archivo WAV y lo convierte a formato matemático interno."""
    import wave
    with wave.open(filepath, 'rb') as wf:
        fs   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return data, fs

def receive_from_file(filepath: str, progress_cb=None) -> dict:
    """Ejecuta la demodulación a partir de una señal guardada en disco (off-line)."""
    if progress_cb: progress_cb(f"Cargando archivo {filepath}...")
    audio, fs = load_modulated_wav(filepath)

    if progress_cb: progress_cb(f"Demodulando {len(audio)/fs:.1f}s de audio...")
    bits = audio_to_bits(audio, fs)

    if progress_cb: progress_cb("Decodificando trama...")
    result = decode_frame(bits)
    result["audio"] = audio
    result["fs"] = fs
    return result

def transmit(filepath: str, use_fec: bool = True, progress_cb=None) -> dict:
    """Interconecta el archivo físico, el modulador y el hardware de salida (parlante)."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if progress_cb: progress_cb(f"Construyendo trama MAC para {len(data)} bytes...")
    audio = build_frame(data, use_fec=use_fec)
    duration = len(audio) / FS

    if progress_cb: progress_cb(f"Transmitiendo {len(audio)} muestras ({duration:.1f}s)...")
    sd.play(audio.astype(np.float32), samplerate=FS)
    sd.wait()

    baud_rate_actual = BAUD_RATE * (7/14 if use_fec else 1)

    return {
        "bytes_sent": len(data),
        "duration_s": duration,
        "baud_rate": baud_rate_actual,
        "audio": audio,
        "fs": FS,
    }

def receive(duration_s: float = 30.0, progress_cb=None) -> dict:
    """Ejecuta la captura desde el hardware de entrada (micrófono) y demodula on-line."""
    if progress_cb: progress_cb(f"Grabando {duration_s}s de canal acústico...")

    n_samples = int(duration_s * FS)
    audio = sd.rec(n_samples, samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    if progress_cb: progress_cb("Demodulando señal cruda...")
    bits = audio_to_bits(audio, FS)

    if progress_cb: progress_cb("Extrayendo payload...")
    result = decode_frame(bits)
    result["audio"] = audio
    result["fs"] = FS
    return result

# =============================================================================
# Herramientas Analíticas
# =============================================================================

def compute_ber_curve(snr_db_range: np.ndarray, n_bits: int = 10000) -> dict:
    """
    Calcula la curva de Tasa de Error de Bit (BER) en función de la SNR.

    Parameters
    ----------
    snr_db_range : np.ndarray
        Vector de valores de relación Señal a Ruido (SNR) a evaluar en dB.
    n_bits : int, opcional
        Cantidad de bits de prueba para la simulación Monte Carlo.

    Returns
    -------
    dict
        Contiene los vectores SNR, BER teórica y BER simulada empíricamente.
    """
    ber_simulated  = []
    ber_theoretical = []

    for snr_db in snr_db_range:
        snr_linear = 10 ** (snr_db / 10)
        
        # Ecuación teórica de BER para detección BFSK no coherente ortogonal
        ber_th = 0.5 * np.exp(-snr_linear / 2)
        ber_theoretical.append(ber_th)

        # Simulación Monte Carlo
        bits_tx = np.random.randint(0, 2, n_bits)
        noise_std = 1.0 / np.sqrt(2 * snr_linear)
        noise = noise_std * np.random.randn(n_bits)
        
        # Umbral de decisión simple en presencia de ruido
        bits_rx = ((bits_tx.astype(float) + noise) > 0.5).astype(int)
        ber_sim = np.mean(bits_tx != bits_rx)
        
        # Piso de error artificial para evitar ceros en representación logarítmica
        ber_simulated.append(max(ber_sim, 1e-6))

    return {
        "snr_db": snr_db_range,
        "ber_simulated": np.array(ber_simulated),
        "ber_theoretical": np.array(ber_theoretical),
    }
