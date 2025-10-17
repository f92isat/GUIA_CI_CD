import os

# Comando para servir el modelo con MLflow
command = "mlflow models serve -m /path/to/mlruns/.../artifacts/model -p 5001 --no-conda"

print("🚀 Desplegando modelo localmente en el puerto 5001...")
print(f"Comando: {command}")

# Ejecutar el comando
os.system(command)