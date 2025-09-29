import papermill as pm

print("Ejecutando pipeline...")

# Paso 1: Ingesta
print("Ejecutando Ingesta...")
pm.execute_notebook(
    'Ingesta.ipynb',
    'Ingesta.ipynb'
)

# Paso 2: EDA
print("Ejecutando EDA...")
pm.execute_notebook(
    'EDA.ipynb',
    'EDA.ipynb'
)

# Paso 3: Limpieza
print("Ejecutando Limpieza...")
pm.execute_notebook(
    'Limpieza.ipynb',
    'Limpieza.ipynb'
)

print("Pipeline completado.")
