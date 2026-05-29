# Proyecto 2: Uso de herramientas de programación para simulación de modulaciones de Banda Lateral Única con el enfoque de la Transformada de Hilberth e implementación de modulaciones digitales pasobanda

## Objetivos

- Diseñar e implementar un simulador de modulación/demodulación de banda lateral única y banda lateral independiente con el enfoque de la transformada de Hilberth por medio de herramientas de programación estructurado.
- Estudiar la transformada de Hilberth para la generación de modulaciones espectralmente eficientes.
- Evaluar el efecto de demodulación de señales SSB síncronas con error de fase y frecuencia.
- Utilizar herramientas de programación estructurada y métodos numéricos para el análisis de sistemas de comunicaciones.
- Envío y recepción de datos digitales utilizando modulaciones pasobanda.

---

# Labor por realizar

Crear una aplicación en Python o Java (o el lenguaje de su elección) que reciba un archivo de audio en formato WAV y realice una modulación y demodulación en banda lateral única a una frecuencia específica.

En la misma aplicación, crear una interfaz que permita el envío de un archivo digital utilizando una modulación pasobanda.

El envío de datos se hará por medio del parlante de la PC (TX) y la recepción será por medio del micrófono de la PC (RX), por lo que las frecuencias a trabajar deben ser compatibles con estos dispositivos. También deberá diseñar un protocolo adecuado para el envío y recepción de los datos.

---

# Datos de entrada

## Modulación SSB

La aplicación deberá solicitar al usuario:

- Archivo de audio en formato WAV.
- Frecuencia de portadora.
- Tipo de modulación SSB:
  - SSB-FC
  - SSB-SC
  - ISB
- Error de fase.
- Error de frecuencia.
- Especificación de cuál banda lateral se desea enviar:
  - USB
  - LSB

### Caso especial ISB

Para el caso de ISB, se deben solicitar dos archivos WAV al usuario.

---

## Modulación digital pasobanda

La aplicación deberá solicitar:

- Cualquier archivo digital.
- Si se utilizará o no algoritmo de corrección de errores.
- Tipo de modulación digital.
- Algoritmo de corrección de errores.

> El tipo de modulación y algoritmo de corrección de errores queda a criterio del estudiante.

---

# Datos de salida

## Modulación SSB

### Modulador

Debe mostrar:

- Gráficas en el dominio del tiempo y frecuencia del mensaje (señal de audio).
- Gráficas de la señal DSB.
- Gráficas de señales SSB e ISB.

### Demodulador

Debe mostrar:

- Gráficas en el dominio del tiempo y frecuencia del mensaje recuperado.
- Reproducción del mensaje de audio recuperado.

---

## Modulación digital pasobanda

### Modulador

Debe mostrar:

- Gráficas en el dominio del tiempo y frecuencia.
- Constelación.

### Demodulador

Debe mostrar:

- Gráficas en el dominio del tiempo y frecuencia del mensaje recuperado.
- Gráfica del BER.
- Diagrama de ojo.
- Tasa de transferencia.

---

# Especificaciones del sistema

## Modulación SSB

1. Formato del archivo de audio de entrada: `WAV`
2. Formato del archivo de audio de salida: `WAV`
3. Frecuencia de portadora máxima: `25 kHz`
4. Error de fase: hasta `180°`
5. Error de frecuencia: `±25%`

## Modulación digital

Las especificaciones quedan a criterio del estudiante, de tal forma que se cumpla el objetivo buscado: el envío y recepción correctos de un archivo digital.

---

# Entregables

## Avance 1 — Semana 13 — 5%

### Debe incluir

- Carga del archivo `.WAV`
- Implementación inicial del modulador
- Revisión en clase

---

## Avance 2 — Semana 14 — 5%

### Debe incluir

- Interfaz según las especificaciones
- Modulador completo
- Implementación inicial del demodulador
- Revisión en clase

---

## Informe Final — Semana 16 — 10%

### Debe incluir

- Informe escrito en formato IEEE detallando:
  - Diseño del sistema
  - Resultados obtenidos
  - Análisis de resultados
- Toda decisión debe estar completamente justificada.
- Se evaluarán habilidades de comunicación escrita.

### Secciones requeridas

- Introducción
- Objetivos
- Marco teórico
- Diseño
- Resultados
- Análisis de resultados
- Conclusiones
- Recomendaciones de mejora
- Referencias
- Bitácoras

### Formato de entrega

- PDF

### Importante

- Debe especificarse claramente las limitaciones del sistema propuesto.

---

## Presentación — Semana 16 — 5%

### Requisitos

- Defensa de la propuesta ante el grupo.
- Tiempo de exposición:
  - 15 minutos
  - Más tiempo de preguntas

### Entregable

- PDF de la presentación.

---

# Administración del proyecto

Se debe generar evidencia sobre el proceso de administración del proyecto:

- Bitácoras de reuniones
- Cronogramas de planeamiento y seguimiento
- Desglose de tareas
- Asignación de responsables
- Designación de un coordinador del grupo

---

# Recomendación

Se recomienda utilizar diagramas de Gantt para la administración y seguimiento del proyecto.
