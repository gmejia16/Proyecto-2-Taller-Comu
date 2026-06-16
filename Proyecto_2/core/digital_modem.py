"""
digital_modem.py
Módulo de modulación/demodulación digital pasobanda (FSK binaria sobre audio).
Protocolo de trama propio para envío de archivos por parlante/micrófono.
EL5522 - Taller de Comunicaciones Eléctricas
"""

import numpy as np
from scipy.signal import butter, filtfilt
import sounddevice as sd
import struct
import hashlib
import time


# ─────────────────────────────────────────────
# Parámetros del sistema
# ─────────────────────────────────────────────

FS        = 44100       # frecuencia de muestreo [Hz]
F_MARK    = 1200        # frecuencia "1" [Hz]
F_SPACE   = 2200        # frecuencia "0" [Hz]
BAUD_RATE = 300         # baudios (símbolos/seg)
SPB       = FS // BAUD_RATE   # muestras por bit
PREAMBLE  = b'\xAA\xAA\xAA\xAA'   # 4 bytes de sincronía
SYNC_WORD = b'\x16\x16\xF0\xF0'   # palabra de sincronía


# ─────────────────────────────────────────────
# Codificación de errores (Hamming 7,4)
# ─────────────────────────────────────────────

# Matriz generadora Hamming(7,4) - forma sistemática
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

# Matriz de paridad
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=np.uint8)


def hamming_encode(nibble: np.ndarray) -> np.ndarray:
    """Codifica un nibble (4 bits) → 7 bits Hamming."""
    return (nibble @ G) % 2


def hamming_decode(codeword: np.ndarray) -> tuple[np.ndarray, bool]:
    """Decodifica 7 bits Hamming → 4 bits. Retorna (datos, corregido)."""
    syndrome = (H @ codeword) % 2
    error_pos = syndrome[0]*4 + syndrome[1]*2 + syndrome[2]*1
    corrected = False
    if error_pos != 0:
        codeword = codeword.copy()
        codeword[error_pos - 1] ^= 1
        corrected = True
    return codeword[:4], corrected


def encode_bytes(data: bytes) -> np.ndarray:
    """Aplica Hamming(7,4) a todos los bytes."""
    bits_out = []
    for byte in data:
        nibble_hi = np.array([(byte >> (7 - i)) & 1 for i in range(4)], dtype=np.uint8)
        nibble_lo = np.array([(byte >> (3 - i)) & 1 for i in range(4)], dtype=np.uint8)
        bits_out.extend(hamming_encode(nibble_hi).tolist())
        bits_out.extend(hamming_encode(nibble_lo).tolist())
    return np.array(bits_out, dtype=np.uint8)


def decode_bits(bits: np.ndarray) -> tuple[bytes, int]:
    """Decodifica bits Hamming(7,4) → bytes."""
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
        byte_val = 0
        for b in hi: byte_val = (byte_val << 1) | int(b)
        for b in lo: byte_val = (byte_val << 1) | int(b)
        result.append(byte_val)
    return bytes(result), corrections


# ─────────────────────────────────────────────
# Modulación BFSK
# ─────────────────────────────────────────────

def bits_to_audio(bits: np.ndarray, fs: int = FS,
                  f1: float = F_MARK, f0: float = F_SPACE,
                  spb: int = SPB) -> np.ndarray:
    """Convierte un array de bits a señal BFSK."""
    t_bit = np.arange(spb) / fs
    audio = []
    for bit in bits:
        freq = f1 if bit == 1 else f0
        tone = np.sin(2 * np.pi * freq * t_bit)
        # Envelope suave para reducir clicks
        window = np.hanning(spb)
        audio.append(tone * np.sqrt(window))
    return np.concatenate(audio)


def build_frame(data: bytes, use_fec: bool = True) -> np.ndarray:
    """
    Construye la trama completa:
    PREAMBLE | SYNC | LONG(4B) | FEC_FLAG(1B) | SHA256(4B) | PAYLOAD
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


# ─────────────────────────────────────────────
# Demodulación BFSK (detección por energía)
# ─────────────────────────────────────────────

def bandpass(signal: np.ndarray, fc: float, bw: float, fs: int) -> np.ndarray:
    """Filtro pasa-banda."""
    nyq = fs / 2
    low  = max((fc - bw/2) / nyq, 0.001)
    high = min((fc + bw/2) / nyq, 0.999)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)


def audio_to_bits(audio: np.ndarray, fs: int = FS,
                  f1: float = F_MARK, f0: float = F_SPACE,
                  spb: int = SPB) -> np.ndarray:
    """Demodula BFSK: detecta bits por energía en cada ventana."""
    bw = abs(f1 - f0) * 0.8          # abs() evita bw negativo si f1 < f0
    sig_mark  = bandpass(audio, f1, bw, fs)
    sig_space = bandpass(audio, f0, bw, fs)

    bits = []
    n_bits = len(audio) // spb
    for i in range(n_bits):
        s = i * spb
        e_mark  = np.sum(sig_mark[s:s+spb] ** 2)
        e_space = np.sum(sig_space[s:s+spb] ** 2)
        bits.append(1 if e_mark > e_space else 0)
    return np.array(bits, dtype=np.uint8)


def sync_bits(bits: np.ndarray) -> int:
    """Busca el inicio de trama por correlación con el SYNC_WORD."""
    sync_bits_ref = np.unpackbits(np.frombuffer(SYNC_WORD, dtype=np.uint8))
    preamble_bits = np.unpackbits(np.frombuffer(PREAMBLE, dtype=np.uint8))
    search = np.concatenate([preamble_bits, sync_bits_ref])
    n = len(search)

    best_score = -1
    best_idx   = 0

    # Limitar búsqueda a los primeros 2000 bits: el preámbulo siempre
    # está al inicio. Buscar en TODO el vector causa O(n²) en archivos grandes.
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
    """Intenta decodificar una trama desde un array de bits."""
    start = sync_bits(bits)
    header_bits = bits[start:start + 9*8]

    if len(header_bits) < 72:
        return {"success": False, "error": "Trama incompleta"}

    header_bytes = np.packbits(header_bits).tobytes()
    length   = struct.unpack('>I', header_bytes[0:4])[0]
    use_fec  = header_bytes[4] == 0x01
    checksum = header_bytes[5:9]

    # Validar que length sea razonable (máx 1 MB)
    if length == 0 or length > 1_000_000:
        return {"success": False, "error": f"Length inválido: {length}"}

    payload_start  = start + 9*8
    n_payload_bits = length * 14 if use_fec else length * 8
    payload_bits   = bits[payload_start:payload_start + n_payload_bits]

    # Validar que haya suficientes bits
    if len(payload_bits) < n_payload_bits:
        return {"success": False,
                "error": f"Payload incompleto: {len(payload_bits)}/{n_payload_bits} bits. "
                          f"Aumentá la duración de grabación."}

    if use_fec:
        data, corrections = decode_bits(payload_bits)
    else:
        data = np.packbits(payload_bits).tobytes()[:length]
        corrections = 0

    expected_checksum = hashlib.sha256(data).digest()[:4]
    ok = (expected_checksum == checksum)

    return {
        "success":      ok,
        "data":         data if ok else None,
        "length":       length,
        "use_fec":      use_fec,
        "corrections":  corrections,
        "checksum_ok":  ok,
        "error":        None if ok else "Checksum no coincide — señal corrupta",
    }




# ─────────────────────────────────────────────
# Utilidades: guardar / cargar audio modulado
# ─────────────────────────────────────────────

def add_awgn(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Agrega ruido blanco gaussiano aditivo (AWGN) a la señal.
    snr_db: relación señal a ruido deseada en dB (0 = muy ruidoso, 40 = casi limpio).
    """
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio.copy()
    snr_linear  = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise       = np.sqrt(noise_power) * np.random.randn(len(audio))
    return np.clip(audio + noise, -1.0, 1.0)


def save_modulated_wav(filepath: str, audio: np.ndarray, fs: int = FS) -> None:
    """Guarda el audio BFSK modulado en un archivo WAV (float32 → int16)."""
    import wave, struct as _struct
    audio_norm = np.clip(audio, -1.0, 1.0)
    samples_int16 = (audio_norm * 32767).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           # int16 = 2 bytes
        wf.setframerate(fs)
        wf.writeframes(samples_int16.tobytes())


def load_modulated_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Carga un WAV de audio BFSK y retorna (audio_float32, fs)."""
    import wave
    with wave.open(filepath, 'rb') as wf:
        fs   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return data, fs


def receive_from_file(filepath: str, progress_cb=None) -> dict:
    """Demodula un archivo WAV de audio BFSK (sin usar micrófono)."""
    if progress_cb:
        progress_cb(f"Cargando archivo {filepath}...")
    audio, fs = load_modulated_wav(filepath)

    if progress_cb:
        progress_cb(f"Demodulando {len(audio)/fs:.1f}s de audio...")
    bits = audio_to_bits(audio, fs)

    if progress_cb:
        progress_cb("Decodificando trama...")
    result = decode_frame(bits)
    result["audio"] = audio
    result["fs"] = fs
    return result



def transmit(filepath: str, use_fec: bool = True,
             progress_cb=None) -> dict:
    """Lee un archivo y lo transmite por el parlante."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if progress_cb:
        progress_cb(f"Construyendo trama para {len(data)} bytes...")

    audio = build_frame(data, use_fec=use_fec)
    duration = len(audio) / FS

    if progress_cb:
        progress_cb(f"Transmitiendo {len(audio)} muestras ({duration:.1f}s)...")

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
    """Escucha por el micrófono y decodifica la trama."""
    if progress_cb:
        progress_cb(f"Grabando {duration_s}s de audio...")

    n_samples = int(duration_s * FS)
    audio = sd.rec(n_samples, samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    if progress_cb:
        progress_cb("Demodulando...")

    bits = audio_to_bits(audio, FS)

    if progress_cb:
        progress_cb("Decodificando trama...")

    result = decode_frame(bits)
    result["audio"] = audio
    result["fs"] = FS
    return result


# ─────────────────────────────────────────────
# Análisis de BER
# ─────────────────────────────────────────────

def compute_ber_curve(snr_db_range: np.ndarray,
                      n_bits: int = 10000) -> dict:
    """Calcula curva BER simulada vs SNR para BFSK no-coherente."""
    ber_simulated  = []
    ber_theoretical = []

    for snr_db in snr_db_range:
        snr_linear = 10 ** (snr_db / 10)
        # BER teórico BFSK no-coherente: 0.5*exp(-Eb/2N0)
        ber_th = 0.5 * np.exp(-snr_linear / 2)
        ber_theoretical.append(ber_th)

        # BER simulado
        bits_tx = np.random.randint(0, 2, n_bits)
        noise_std = 1.0 / np.sqrt(2 * snr_linear)
        noise = noise_std * np.random.randn(n_bits)
        # Decisión simple con ruido
        bits_rx = ((bits_tx.astype(float) + noise) > 0.5).astype(int)
        ber_sim = np.mean(bits_tx != bits_rx)
        ber_simulated.append(max(ber_sim, 1e-6))

    return {
        "snr_db": snr_db_range,
        "ber_simulated": np.array(ber_simulated),
        "ber_theoretical": np.array(ber_theoretical),
    }
