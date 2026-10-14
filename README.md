# Clasificación de lesiones de córnea basado en redes neuronales CNN aplicando técnicas de pseudocolor

## 1. Qué hace el proyecto

El proyecto desarrolla un sistema de **clasificación binaria** de imágenes de tomografía de coherencia óptica (OCT)
que distingue entre tejido **normal** y tejido con **lesión**. Para lograrlo se combinan dos ideas:

- **Pseudocolor:** las imágenes OCT son en escala de grises. Se convierten a color asignando un color a cada nivel de
  intensidad, y se evalúa si eso ayuda a la red a clasificar mejor.
- **Redes neuronales convolucionales (CNN) con transferencia de aprendizaje:** se parte de redes ya entrenadas con
  millones de imágenes y se adaptan a este problema.

### Etapas del proyecto

1. **Preparar los datos.** Consolidar el conjunto de imágenes OCT y establecer las dos clases: normal y lesión.
2. **Evaluar el pseudocolor.** Probar al menos tres técnicas de pseudocolor, empezando por tonos fríos y ampliando a
   otras representaciones si los resultados lo justifican.
3. **Entrenar las redes.** Entrenar ResNet, DenseNet y MobileNet con transferencia de aprendizaje, usando las imágenes
   originales en gris como referencia.
4. **Comparar las arquitecturas.** Compararlas con el pseudocolor de mejores resultados, usando accuracy, precision,
   recall, F1-score, especificidad, ROC-AUC y matriz de confusión.
5. **Fine-tuning.** Ajustar las dos mejores arquitecturas, probando distintos niveles de adaptación de sus capas.
6. **Elegir la configuración final.** Buscar el mejor equilibrio entre desempeño y eficiencia: tiempo de inferencia,
   consumo de memoria y uso de recursos.
7. **Implementar en el dispositivo embebido.** Evaluar su funcionamiento, tiempo de inferencia y consumo de recursos.

### Cómo se hace la comparación

En cada experimento se cambia una sola cosa (el filtro de color o la arquitectura) y se mantiene todo lo demás igual:
mismos datos, misma división y misma configuración de entrenamiento. Así, la diferencia en los resultados se puede
atribuir a lo que se cambió.

El pseudocolor no agrega información nueva a la imagen: solo reparte la intensidad en tres canales (R, G, B). Por eso
todos los filtros se comparan contra una **línea base en gris** (`gray3`: la misma imagen en gris copiada en los tres
canales).

## 2. Origen de los datos y del código

| | Fuente |
|---|---|
| Dataset | OCTDL, versión 4: https://data.mendeley.com/datasets/sncdhf53xc/4 |
| Código base | https://github.com/MikhailKulyabin/OCTDL |

El dataset tiene **2064 imágenes OCT de 820 pacientes**, etiquetadas en 7 categorías. Para este proyecto se agrupan
en dos clases:

| Clase | Categorías originales | Imágenes |
|---|---|---|
| `normal` | NO | 332 (16 %) |
| `lesion` | AMD, DME, ERM, RAO, RVO, VID | 1732 (84 %) |

Del código base se conserva el entrenamiento y la evaluación de la red. Lo que se agregó o modificó en este proyecto:

- `make_binary.py`: convierte las 7 etiquetas originales en las 2 clases para la clasificacion binaria.
- `preprocessing.py`: divide los datos en las 2 clases, con una división fija y reproducible.
- `pseudocolor.py`: aplica los filtros de pseudocolor.
- `gradcam.py`: muestra en qué zonas de la imagen se fija la red.

## 3. Cómo funciona, paso a paso desde 0 

```
Dataset OCTDL ──> 1. Etiquetas binarias ──> 2. División train/val/test ──> 3. Pseudocolor ──> 4. Entrenamiento ──> 5. Grad-CAM
                     make_binary.py            preprocessing.py              pseudocolor.py     main.py              gradcam.py
```

### Paso 1. Etiquetas binarias (`make_binary.py`)

Lee el archivo de etiquetas original y marca cada imagen como `normal` (categoría NO) o `lesion` (cualquier otra).

### Paso 2. División de los datos (`preprocessing.py`)

Separa las imágenes en entrenamiento (60 %), validación (15 %) y prueba (25 %).

La división se hace **por paciente**, no por imagen: todas las imágenes de un mismo paciente quedan en el mismo
conjunto. Esto evita que la red sea evaluada con imágenes de un paciente que ya vio al entrenar, lo que daría
resultados artificialmente altos. La semilla es fija (def. 42), así que la división siempre es la misma.

| Conjunto | normal | lesion | Total |
|---|---|---|---|
| Entrenamiento | 200 | 1041 | 1241 |
| Validación | 46 | 256 | 302 |
| Prueba | 86 | 435 | 521 |

### Paso 3. Pseudocolor (`pseudocolor.py`)

Toma el dataset ya dividido y genera una copia con el filtro aplicado. A cada imagen se le hace lo siguiente:

1. **Filtro de mediana** (3×3): reduce el ruido granulado del OCT.
2. **CLAHE**: mejora el contraste local.
3. **Mapa de color**: convierte cada nivel de gris en un color.

Filtros disponibles:

| Tipo | Filtros |
|---|---|
| Tonos fríos | `winter`, `ocean`, `cool` |
| Tonos cálidos | `hot`, `autumn`, `inferno` |
| Referencia | `jet`, `viridis` |
| Línea base | `gray3` |
| Propio | `oct_custom`: sin probar aun

Como el filtro se aplica sobre el dataset ya dividido, todos los filtros usan exactamente las mismas imágenes en
entrenamiento, validación y prueba.

### Paso 4. Entrenamiento y evaluación (`main.py`)

Se usa **transfer learning**: una red **ResNet50** preentrenada, que se ajusta para las 2 clases.

| Parámetro | Valor |
|---|---|
| Tamaño de entrada | 224 × 224 |
| Épocas | 100 |
| Tamaño de lote | 64 |
| Optimizador | Adam |
| Métricas | exactitud, F1, AUC, precisión, sensibilidad |

Como hay muchas más imágenes con lesión que normales, se usa muestreo balanceado y pesos por clase en la pérdida.

Durante el entrenamiento se evalúa en validación en cada época y se guarda el modelo con mejor AUC. Al final, ese
modelo se evalúa una sola vez en el conjunto de prueba. Toda la configuración está en `configs/OCTDL.yaml`.

### Paso 5. Interpretación (`gradcam.py`)

Genera mapas de calor **Grad-CAM** sobre imágenes de prueba, que señalan las zonas en las que la red se apoyó para
decidir. Se generan cuatro grupos: normales y lesiones bien clasificadas, falsos positivos y falsos negativos. Sirve
para revisar si la red mira la zona de la lesión o detalles irrelevantes.

## 4. Cómo ejecutarlo

```bash
pip install -r requirements.txt

# 1. Dataset
Descargar el dataset de Medeley e ingresarlo en la carpeta data

# 2. Etiquetas binarias
python make_binary.py data/OCTDL_labels.csv data/OCTDL_binary_labels.csv

# 3. División train/val/test  ->  data/OCTDL_procesado
python preprocessing.py

# 4. Dataset con filtro  ->  data/OCTDL_pseudocolor
python pseudocolor.py --name winter

# 5. Entrenar y evaluar (usar la carpeta indicada en data_path de configs/OCTDL.yaml)
python main.py

# 6. Mapas Grad-CAM  ->  gradcam_out/ (en data_path debe indicarse el mismo dataset con el que se entreno)
python gradcam.py --weights ./run/best_validation_weights.pt
```

Para comparar otro filtro se repiten los pasos 4 y 5 cambiando `--name` (por ejemplo `--name viridis` para la línea
base). El dataset original se descarga del enlace de Mendeley y se coloca en `data/OCTDL/`.

## 5. Estructura del proyecto

```
configs/OCTDL.yaml    configuración del entrenamiento
data/                 datasets y código de carga de imágenes
modules/              construcción de la red
utils/                métricas y funciones auxiliares
make_binary.py        paso 1
preprocessing.py      paso 2
pseudocolor.py        paso 3
main.py, train.py     paso 4
gradcam.py            paso 5
```

## 6. Referencia

Kulyabin, M., Zhdanov, A., Nikiforova, A., et al. *OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep
Learning Methods*. Scientific Data 11, 365 (2024). https://doi.org/10.1038/s41597-024-03182-7
