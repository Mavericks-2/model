import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import numpy as np
from joblib import load
import os
from modelController import getClassification

# Cargar el modelo
# dt = load("HousesRandomForest.joblib")

# Generar el servidor (Back End)
servidorWeb = Flask(__name__)

# Enable CORS for all routes
CORS(servidorWeb, resources={r"/*": {"origins": "*"}})

# productMatrixCatalog = {
#     1: "Takis Fuego 80g.",
#     2: "Takis Original 80g.",
#     3: "Runners 80g.",
#     4: "Chips Jalapeño 60g.",
#     5: "Chips Fuego 60g.",
#     6: "Tostitos 57g.",
#     7: "Cheetos Torciditos 44g.",
#     8: "Fritos Limón y Sal 38g.",
#     9: "Churrumais",
#     10: "Rancheritos 40g.",
#     11: "Sabritas Sal 36g.",
#     12: "Cheetos Flamin Hot 44g.",
#     13: "Doritos Nacho 48g.",
#     14: "Pop Karameladas 120g.",
#     15: "Hot Nuts Original 160g.",
#     16: "Bitz Cacahuate Enchilado 90g.",
#     17: "Bitz Almendras con Sal 32g.",
#     18: "Bitz Cacahuates Enchilados 95g.",
#     19: "Leo Mix Botanero 80g.",
#     20: "Maruchan Pollo con Vegetales 64g.",
#     21: "Botanera Chilito 125g.",
#     22: "Tajín Dulce 160g.",
#     23: "Salsa Búfalo Clásica 150g.",
#     24: "Del Primo Salsa Guacamole 300g.",
#     25: "Nestle La Lechera Original 335g.",
#     26: "Nestle Carnation Leche Evaporada 360g.",
#     27: "Chips Papatinas 90g.",
#     28: "Ruffles Queso 41g.",
#     29: "Maruchan Carne de Res 64g.",
#     30: "Nissin Camarón Picante 64g.",
#     31: "Nissin Carne de Res 64g.",
#     32: "Bitz Cacahuate Habanero 110g.",
#     33: "Semillas de Girasol 70g.",
#     34: "Cacahuates Sal Bokados 90g.",
#     35: "Cacahuates Japonés Leo 90g.",
#     36: "Semillas de Calabaza Bokados 30g."
# }

# Catalogo para el modelo de 8 productos
productMatrixCatalog = {
    1: 'Cheetos Torciditos',
    2: 'Chips Fuego',
    3: 'Chips Jalapeño',
    4: 'Hut Nuts',
    5: 'Maruchan PolloConVegetales',
    6: 'Nissin CamaronPicante',
    7: 'Takis Fuego',
    8: 'Tostitos'
}


def dataProcessing(infoData):
    pass


def obtainProduct(imagen_recortada):
     # Hacer resize a la imagen 256x256
    imagen_recortada = imagen_recortada.resize((256, 256))
    # Obtener la clasificación
    classification = getClassification(imagen_recortada)
    return int(classification)


def getPlanogramScheme(coordinates):
    #  Ordenar las coordenadas por y
    coordinates.sort(key=lambda x: x["y"])

    # Definir diccionario
    result = {}
    for coordinate in coordinates:
        if coordinate["y"] in result:
            result[coordinate["y"]].append(coordinate)
        else:
            result[coordinate["y"]] = []
            result[coordinate["y"]].append(coordinate)

    # Por cada fila, ordenar por x
    for row in result:
        result[row].sort(key=lambda x: x["x"])

    #  Generar el planograma final [[{}]]
    result = list(result.values())

    return result


def getPlanogramProducts(planogram, image):
    # Definir arreglo de productos
    products = []
    #  Obtener los productos de cada fila
    for row in planogram:
        rowProducts = []
        for product in row:
            #  Obtener las coordenadas
            x = product["x"]
            y = product["y"]
            width = product["width"]
            height = product["height"]
            # Recorta la imagen
            imagen_recortada = image.crop((x, y, x + width, y + height))
            #  Obtener el producto
            product = obtainProduct(imagen_recortada)
            # Agregar el producto a la fila
            rowProducts.append(product)
        # Agregar la fila a los productos
        products.append(rowProducts)

    return products

@servidorWeb.route("/compareImages", methods=["POST"])
def compare():
    resultMatrix = []
    row_index = 0
    column_index = 0

    planogramMatrix = request.json["data"]["planogram"]
    photoMatrix = request.json["data"]["actualPlanogram"]

    planogramMatrix = planogramMatrix["coordenadas"]
    photoMatrix = photoMatrix["coordenadas"]

    print("Planogram Matrix: ", planogramMatrix)
    print("Photo Matrix: ", photoMatrix)

    for row in planogramMatrix:
        column_index = 0
        for product in row:
            is_correct = True
            if product != photoMatrix[row_index][column_index]:
                is_correct = False
            resultMatrix.append({
                "row": row_index,
                "column": column_index,
                "currentProduct": product,
                "expectedProduct": photoMatrix[row_index][column_index],
                "isCorrect": is_correct
            })
            column_index += 1
        row_index += 1

    print("Result Matrix: ", resultMatrix)

    return jsonify({"resultMatrix": resultMatrix})


@servidorWeb.route("/classifyImage", methods=["POST"])
def classify():
    if "coordenadas" not in request.json["data"]:
        return "No image part in the form"

    rectangles = request.json["data"]["coordenadas"]

    # Obtener la imagen actual
    image = Image.open("imagenActual/imagenActual.jpg")  # Obtenerla de la bd

    #  Obtener el esquema del planograma
    scheme = getPlanogramScheme(rectangles["coordenadas"])
    planogram = getPlanogramProducts(scheme, image)

    print("Planogram: ", planogram)
    return planogram


@servidorWeb.route("/uploadImage", methods=["POST"])
def upload():
    base64_data = request.json["imagen"]
    image_data = base64.b64decode(base64_data)
    imagen = Image.open(io.BytesIO(image_data))
    imagen.save("imagenActual/imagenActual.jpg")

    return {"message": "ok"}


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
    servidorWeb.run(debug=False, host="0.0.0.0", port="8083")
