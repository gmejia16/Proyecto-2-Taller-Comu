# EL5522 - Taller de Comunicaciones Eléctricas
## Simulador de Modulación SSB y Módem Digital Pasobanda

Este repositorio contiene la implementación de una aplicación con interfaz gráfica (GUI) diseñada para simular, analizar y ejecutar transmisiones reales de modulaciones analógicas de Banda Lateral Única (SSB) y modulaciones digitales pasobanda (BFSK). 

El sistema permite el envío físico de datos digitales y señales de audio a través del hardware de la computadora (micrófono y parlante), aplicando conceptos de procesamiento digital de señales (DSP) y la Transformada de Hilbert.

---

## Características Principales

### Modulador y Demodulador Analógico (SSB / ISB)
- **Método de Hilbert:** Generación espectralmente eficiente de señales de banda única.
- **Modos Soportados:** SSB-SC (Portadora Suprimida), SSB-FC (Portadora Completa) e ISB (Banda Lateral Independiente con doble archivo de audio).
- **Selección de Banda:** Transmisión configurable en banda lateral superior (USB) o inferior (LSB).
- **Simulación de Canal Receptivo:** Inyección de error de fase (hasta 180°) y error de frecuencia (±25%) para evaluar el desempeño de la demodulación síncrona.
- **Portadora de Alta Frecuencia:** Soporte para portadoras de hasta 25 kHz con técnicas de sobremuestreo (resampling) para evitar aliasing.

### Módem Digital Pasobanda (Audio FSK)
- **Transmisión Acústica:** Envío y recepción de archivos de datos a través del parlante (TX) y micrófono (RX).
- **Esquema de Modulación:** BFSK (Binary Frequency Shift Keying) optimizado para la banda de audio.
- **Protocolo de Trama Personalizado:** Implementación de preámbulo, palabra de sincronización, longitud de carga útil y validación criptográfica (SHA-256).
- **Corrección de Errores (FEC):** Codificación de canal tipo Hamming (7,4) seleccionable por el usuario.
- **Evaluación de Desempeño:** Cálculo de Tasa de Error de Bit (BER) empírica versus teórica y generación de diagrama de ojo.

---

## Arquitectura del Sistema implementada

La lógica fundamental del procesamiento de señales se divide en dos módulos principales:

### 1. Flujo Analógico (SSB por Transformada de Hilbert)
```mermaid
graph TD
    A[Audio WAV Original] --> B[Normalización de Amplitud]
    B --> C{¿Modo de Operación?}
    
    C -- SSB-SC / FC --> D[Cálculo de Transformada de Hilbert]
    D --> E[Señal en Fase]
    D --> F[Señal en Cuadratura]
    
    E --> G[Multiplicar por Portadora Cos]
    F --> H[Multiplicar por Portadora Sin]
    
    G --> I[Señal DSB]
    H --> J[Cancelación de Fase]
    
    I --> K{Banda Lateral}
    J --> K
    
    K -- USB --> L[Resta: DSB - Cuadratura]
    K -- LSB --> M[Suma: DSB + Cuadratura]
    
    C -- ISB --> N[Procesar Audio 1 como USB]
    C -- ISB --> O[Procesar Audio 2 como LSB]
    N --> P[Suma de Bandas Independientes]
    O --> P


graph TD
    A[Archivo de Datos / Bytes] --> B{¿Aplicar FEC?}
    
    B -- Sí --> C[Codificador Hamming 7,4]
    B -- No --> D[Conversión Directa a Bits]
    
    C --> E[Empaquetado de Trama]
    D --> E
    
    E --> F[Generar Preamble y Sync Word]
    F --> G[Cálculo SHA256 Checksum]
    
    G --> H[Modulador BFSK]
    H --> I[Asignación de Tonos]
    I --> J[Mark: 1200 Hz]
    I --> K[Space: 2200 Hz]
    
    J --> L[Transmisión Acústica por Parlante]
    K --> L
