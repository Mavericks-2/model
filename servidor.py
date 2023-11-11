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

def scaleImage(image, width, height):
    #  Obtener el tamaño de la imagen
    image_width, image_height = image.size

    #  Obtener el factor de escala
    scale = min(width / image_width, height / image_height)

    #  Obtener el nuevo tamaño
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    #  Redimensionar la imagen
    image = image.resize((new_width, new_height))

    return image

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

def getLastAdded():
    #  Obtener el último archivo agregado
    lastAdded = 0
    for file in os.listdir("imagenActual/recortes/"):
        if file.endswith(".jpg"):
            lastAdded = int(file.split(".")[0])
    return lastAdded

def getPlanogramProducts(planogram, image):
    # Definir arreglo de productos
    products = []
    lastAdded = getLastAdded()
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
            lastAdded += 1
            # Validate that the name is not already in the folder
            while os.path.exists("imagenActual/recortes/"+str(lastAdded) + ".jpg"):
                lastAdded += 1
            # Guarda la imagen recortada en una carpeta
            imagen_recortada.save("imagenActual/recortes/"+str(lastAdded)+ ".jpg")

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

    return jsonify({"resultMatrix": resultMatrix})


@servidorWeb.route("/classifyImage", methods=["POST"])
def classify():
    if "coordenadas" not in request.json["data"]:
        return "No image part in the form"

    rectangles = request.json["data"]["coordenadas"]

    # Obtener la imagen actual
    image = Image.open("imagenActual/imagenActual.jpg")

    #  Obtener el esquema del planograma
    scheme = getPlanogramScheme(rectangles["coordenadas"])
    planogram = getPlanogramProducts(scheme, image)

    return planogram


@servidorWeb.route("/uploadImage", methods=["POST"])
def upload():
    base64_data = request.json["imagen"]
    image_data = base64.b64decode(base64_data)
    imagen = Image.open(io.BytesIO(image_data))

     

    # erase previous image if exists
    if os.path.exists("imagenActual/imagenActual.jpg"):
        os.remove("imagenActual/imagenActual.jpg")

    # wait for the image to be deleted
    while os.path.exists("imagenActual/imagenActual.jpg"):
        pass

    if request.json["scaleWidth"] > 0:
        imagen = imagen.transpose(Image.ROTATE_270)
        imagen = scaleImage(imagen, request.json["scaleWidth"], request.json["scaleHeight"])
    else: 
        print("The image is not scaled")
        imagen = imagen.transpose(Image.ROTATE_270)

    # Saves images
    imagen.save("imagenActual/imagenActual.jpg")

    # Wait for the image to be saved
    while not os.path.exists("imagenActual/imagenActual.jpg"):
        pass

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

@servidorWeb.route("/getImageSize", methods=["GET"])
def getImageSize():
    image = Image.open("imagenActual/imagenActual.jpg")
    width, height = image.size
    return jsonify({"width": width, "height": height})


if __name__ == "__main__":
    servidorWeb.run(debug=False, host="0.0.0.0", port="8083")
