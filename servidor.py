from flask import Flask, request, jsonify
from PIL import Image
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

@servidorWeb.route("/classifyImage", methods=["POST"])
def classify():
    index = 0
    if "coordenadas" not in request.json:
        return "No image part in the form"

    rectangles = request.json["coordenadas"]

    # Recorre los rectangulos
    for rectangle in rectangles['coordenadas']:
        # Obtiene las coordenadas
        x = rectangle["x"]
        y = rectangle["y"]
        width = rectangle["width"]
        height = rectangle["height"]

        # Abre la imagen sin saber el nombre
        for filename in os.listdir("imagenActual"):
            imagen = Image.open("imagenActual/" + filename)

        # Recorta la imagen
        imagen_recortada = imagen.crop((x, y, x + width, y + height))

        # TODO: AGREGAR FUNCIONALIDAD DE COMPARACION CON EL MODELO
        # Guarda la imagen recortada
        name = "imagen_recortada " + str(index) + ".jpg"
        imagen_recortada.save("croppedImages/" + name)
        index += 1

    return "Image uploaded successfully"


@servidorWeb.route("/uploadImage", methods=["POST"])
def upload():
    if "imagen" not in request.files:
        return "No image part in the form"

    file = request.files["imagen"]

    if file.filename == "":
        return "No selected image"

    # You can process the image here, save it to a directory, or perform other actions.
    # For example, you can save it to the 'uploads' directory:
    # file.save('uploads/' + file.filename)

    # Abre la imagen
    imagen = Image.open(file)

    # Obtiene el width y height de la imagen
    width, height = imagen.size
    # print("Width: ", width)
    # print("Height: ", height)

    imagen.save("imagenActual/" + file.filename)

    return "Image uploaded successfully"


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
    servidorWeb.run(debug=False, host="0.0.0.0", port="8081")
