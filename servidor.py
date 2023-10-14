from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from joblib import load
import os

# Cargar el modelo
# dt = load("HousesRandomForest.joblib")

# Generar el servidor (Back End)
servidorWeb = Flask(__name__)

# Enable CORS for all routes
CORS(servidorWeb, resources={r"/*": {"origins": "*"}})

def dataProcessing(infoData):
    pass

# Envío de datos a través de JSON
@servidorWeb.route("/model/getProductMatrix", methods=["POST"])
def modelo():
    # Recibir los datos de la petición
    infoData = request.json
    # print("Datos: \n", infoData, "\n")

    # Preprocesamiento de los datos
    dataProcessing(infoData)

    # Convertir los datos en un array
    datos_array = np.array(list(infoData.values()))

    # Predecir el valor de la calidad del vino
    # prediccion = dt.predict(datos_array.reshape(1, -1))

    # Retornar la predicción en formato JSON
    return jsonify({"prediccion": "prediccion"})

if __name__ == "__main__":
    servidorWeb.run(debug=False, host="0.0.0.0", port="8080")