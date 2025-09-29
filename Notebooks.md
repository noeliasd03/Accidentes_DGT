# Análisis de Siniestralidad Vial en España
Este proyecto tiene como objetivo el análisis y limpieza de un conjunto de datos sobre accidentes de tráfico en España, combinado con datos de intensidad de tráfico, con el fin de generar un dataset robusto, enriquecido y listo para análisis exploratorio y modelado.

### 1. Ingesta de Datos
El notebook de ingesta se encarga de unir dos fuentes clave:  
- Accidentes de tráfico (en formato Excel).
- Datos de tráfico mensual (extraídos de un PDF de estaciones permanentes).
  
 **Funcionalidades principales**
  
- Conexión y descarga desde Azure Blob Storage.
- Lectura y conversión de datos a Spark DataFrames.
- Extracción automática de datos de PDFs con pdfplumber.
- Transformación y normalización de datos de tráfico.
- Unión entre accidentes y estaciones por carretera y mes, eligiendo la estación más cercana al punto kilométrico.
- Exportación final del dataset enriquecido a formato Parquet.

### 2. Análisis Exploratorio de Datos (EDA)
El EDA permite comprender la estructura, calidad y patrones del dataset procesado.  

**Principales análisis**

- Carga del Parquet desde Azure en formato pandas para mayor flexibilidad en la exploración.
- Revisión de dimensiones, tipos de datos y muestra inicial del dataset.
- Detección de valores nulos y filas duplicadas.
- Análisis temporal (mes, día de la semana, hora del día) y espacial (provincias con mayor siniestralidad).
- Estudio de la gravedad de los accidentes y del tipo de vehículos implicados.
- Evaluación de las condiciones ambientales (climatología, iluminación).
- Cálculo de correlaciones entre variables numéricas clave.
- Visualización de la distribución de variables con boxplots y gráficos de barras.
- Visualización de outliers con boxplots de variables numéricas, sin encontrar valores claramente atípicos.

 ### 3. Limpieza de Datos
La limpieza se realiza con PySpark, garantizando calidad y coherencia en grandes volúmenes de datos. 

**Pasos principales**

1. Carga eficiente desde Azure con Spark.
2. Eliminación de duplicados.
3. Corrección e imputación de la variable ISLA a partir de código postal y prefijo de carretera.
4. Sustitución de nulos en campos como NUDO_INFO, CONDICION_NIEBLA y CONDICION_VIENTO.
5. Conversión de tipos para asegurar integridad.
6. Verificación de datos y valores únicos.
7. En cuanto a los outliers, tras el análisis en el EDA, se decidió no aplicar tratamientos adicionales, ya que:
    - Las variables numéricas son en su mayoría codificaciones categóricas.
    - No se identificaron valores extremos inconsistentes en el contexto de los datos.
8. Exportación final a Parquet consolidado.

***

### Diseño y Ejecución del Pipeline de Datos en Azure ML Studio
**Objetivo**  

Automatizar la ejecución de los tres notebooks principales (Ingesta, EDA y Limpieza) en secuencia, respetando sus entornos de ejecución (PySpark o Python) usando solo las herramientas disponibles en una cuenta gratuita de Azure ML Studio (sin clústeres Spark dedicados).

**Arquitectura del Pipeline**

El flujo está compuesto por los siguientes pasos ejecutados secuencialmente mediante un script pipeline.py:
1. Ingesta.ipynb – Lectura y carga de datos desde Azure Blob Storage utilizando PySpark.
2. EDA.ipynb – Análisis exploratorio con pandas, seaborn y matplotlib (ejecutado en entorno Python).
3. Limpieza.ipynb – Transformación, validación y depuración del dataset (también en PySpark).

**Entornos configurados**

Se han creado dos entornos personalizados en Azure ML Studio para ejecutar cada notebook con las dependencias adecuadas:  
- pyspark-env
    - Basado en un entorno existente compatible con PySpark.
    - Se añadieron librerías específicas como pdfplumber y openpyxl, necesarias para la ingesta de PDFs y Excels.
    - Justificación: los notebooks de ingesta y limpieza requieren procesar grandes volúmenes y estructuras de datos con Spark.
- python-env
    - Basado en un entorno con Python puro, orientado al análisis con pandas, seaborn y matplotlib.
    - Justificación: el notebook de EDA necesita agilidad y herramientas de visualización, que son más eficientes en pandas.

**Ejecución del Pipeline**
1. Se creó un Job en Azure ML Studio que ejecuta pipeline.py dentro del entorno pyspark-env.
2. Azure ML ejecuta el script usando papermill, respetando la secuencia y los entornos de cada notebook.
3. Los outputs son el mismo archivo que el input para no duplicar nuestra estructura y ver las salidas del código en el propio notebook. 

**Ventajas de este enfoque**
- Es compatible con las restricciones de la cuenta gratuita de Azure
- Permite mantener la trazabilidad y visualización clara de cada etapa del proceso.
- Ejecuta el pipeline completo con un solo Job.
- Compatible con distintos entornos y recursos computacionales según la necesidad del paso.

