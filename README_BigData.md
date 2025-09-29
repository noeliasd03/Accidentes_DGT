# Arquitectura del Proyecto en Azure Machine Learning
Este proyecto de análisis de siniestralidad vial está desplegado y ejecutado íntegramente en el ecosistema de Azure Machine Learning Studio, utilizando un diseño modular basado en notebooks y adaptado a las capacidades de una cuenta gratuita.  
A continuación se detalla la arquitectura de recursos, herramientas y configuración utilizada.

### Estructura General en Azure
1. **Grupo de recursos (Resource Group)**
    - Contenedor principal que agrupa todos los servicios relacionados con el proyecto.
    - Permite gestionar el ciclo de vida de los recursos de forma centralizada.
3. **Workspace de Azure Machine Learning**
     - Punto central del proyecto de machine learning.
     - Aloja y coordina notebooks, entornos, datasets, jobs, pipelines, métricas y artefactos.
     - Proporciona UI visual (ML Studio) para gestión sin necesidad de código adicional.
5. **Almacenamiento en Azure Blob**
     - Repositorio principal de archivos.
     - Utilizado para:
         - Contenedor raw-data: Subida de datos de entrada (Excel, PDF).
         - Contenedor processed-data: Almacenamiento de datos intermedios y resultados (.parquet).

***

### Entornos de Ejecución
Se crearon dos entornos personalizados para separar lógica de procesamiento Spark y análisis en Python:
- **pyspark-env** : Basado en un entorno existente compatible con PySpark.
- **python-env** : Basado en un entorno con Python puro, orientado al análisis con pandas, seaborn y matplotlib.

***

### Notebooks y Funciones  
**Mirar documentación específica de los Notebooks y pipeline: Notebooks.md**  
Todos los procesos están implementados como notebooks .ipynb y alojados en el espacio de usuario:  
**Notebooks :**
- Ingesta.ipynb (Pyspark) : Descarga, lectura y combinación de datos desde Azure Blob
- EDA.ipynb (Python) : Análisis exploratorio, visualización de datos, detección visual de outliers
- Limpieza.ipynb (Pyspark) : Validación, limpieza, corrección de campos, export final

***

### Orquestación del Pipeline
Script en Python que usa Papermill para ejecutar notebooks en orden secuencial (Ingesta -> EDA -> Limpieza) :
- Ejecutado como Job en Azure ML Studio.
- Utiliza pyspark-env como entorno base (Papermill puede lanzar notebooks con distintos kernels).
- Los notebooks se sobrescriben para mantener las salidas visibles sin generar duplicados.
