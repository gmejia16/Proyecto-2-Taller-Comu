# EL5522 — Proyecto 2
## Simulador SSB + Modulación Digital Pasobanda

### Estructura del proyecto
```
proyecto2/
│
├── app.py                   ← Punto de entrada (GUI principal)
│
├── core/
│   ├── ssb_modulator.py     ← Modulación/Demodulación SSB e ISB (Hilbert)
│   └── digital_modem.py     ← BFSK pasobanda + Hamming(7,4) + protocolo de trama
│
├── utils/
│   └── plots.py             ← Todas las gráficas (tiempo, frecuencia, BER, ojo)
│
└── requirements.txt
```

---

### Instalación

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# En Linux puede ser necesario:
sudo apt install python3-tk portaudio19-dev

# 2. Ejecutar la aplicación
python app.py
```

---

### Funcionalidades implementadas

#### Pestaña SSB / ISB
- Carga de archivo WAV mono/estéreo
- Modulación SSB-SC, SSB-FC e ISB mediante Transformada de Hilbert
- Selección de banda lateral (USB / LSB)
- Demodulación síncrona con error de fase (0–180°) y error de frecuencia (±25%)
- Gráficas: dominio tiempo y frecuencia de mensaje, DSB, SSB/ISB
- Gráficas del mensaje recuperado (tiempo + espectro)
- Reproducción del audio demodulado
- Guardado del audio recuperado como WAV

#### Pestaña Digital Pasobanda
- Transmisión de cualquier archivo digital por parlante (BFSK, 300 Bd)
- Frecuencias: Mark=1200 Hz, Space=2200 Hz (compatibles con micrófono/parlante)
- Protocolo de trama propio: PREAMBLE + SYNC + LENGTH + FEC_FLAG + SHA256 + PAYLOAD
- Corrección de errores opcional: Hamming(7,4)
- Recepción por micrófono con decodificación automática
- Gráficas: tiempo, espectro, constelación BFSK
- Curva BER teórica vs simulada
- Diagrama de ojo
- Guardado del archivo recibido

---

### Protocolo de trama digital

```
[ 4B PREAMBLE ][ 4B SYNC ][ 4B LENGTH ][ 1B FEC_FLAG ][ 4B SHA256 ][ PAYLOAD ]
  0xAAAAAAAA    0x1616F0F0   uint32 BE      0x01/0x00    checksum      datos
```

- Con FEC activado: cada byte se codifica en 14 bits usando Hamming(7,4)
- Corrección de 1 bit de error por cada grupo de 7 bits

---

### Notas del sistema
- fs máxima de portadora SSB: 25 kHz
- Error de fase: hasta 180°
- Error de frecuencia: hasta ±25%
- Frecuencias BFSK compatibles con hardware de audio estándar
