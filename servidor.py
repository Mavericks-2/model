"""
@authors: Pablo González, José Ángel García, Erika Marlene
@description: Servidor con una colección de rutas
que permiten la comunicación con el cliente, al igual
que la manipulación de las imágenes y la clasificación
de los productos.
"""

import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import numpy as np
import os
from modelController import getClassification

labels = [
    "CheetosTorciditos",
    "ChipsJalapeño",
    "Churrumais",
    "DoritosNachos",
    "FritosLimonYSal",
    "HutNuts",
    "PopKarameladas",
    "Rancheritos",
    "RufflesQueso",
    "Runners",
    "TakisFuego",
    "TakisOriginal",
    "Tostitos",
]

# Generar el servidor (Back End)
servidorWeb = Flask(__name__)

# Enable CORS for all routes
CORS(servidorWeb, resources={r"/*": {"origins": "*"}})

def scaleImage(image, width, height):
    """
    Función que redimensiona una imagen a un tamaño específico
    :param image: Imagen a redimensionar
    :param width: Ancho de la imagen
    :param height: Alto de la imagen
    :return: Imagen redimensionada
    """
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


def obtainProduct(imagen_recortada):
    """
    Función que obtiene la clasificación de un producto
    :param imagen_recortada: Imagen del producto
    :return: Clasificación del producto
    """
    # Hacer resize a la imagen 256x256
    imagen_recortada = imagen_recortada.resize((256, 256))
    # Obtener la clasificación
    classification = getClassification(imagen_recortada)
    return int(classification)


def getPlanogramScheme(coordinates):
    """
    Función que obtiene el esquema del planograma
    :param coordinates: Coordenadas de los productos
    :return: Esquema del planograma
    """
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


def getLastAdded(folderName="imagenActual/recortes/"):
    """
    Función que obtiene el último archivo agregado
    :param folderName: Nombre de la carpeta
    :return: Último archivo agregado
    """
    #  Obtener el último archivo agregado
    lastAdded = 0
    # Evaluar si existe el directorio
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    for file in os.listdir(folderName):
        if file.endswith(".jpg"):
            lastAdded = int(file.split(".")[0])

    return lastAdded


def getPlanogramProducts(planogram, image):
    """
    Función que obtiene los productos del planograma
    :param planogram: Esquema del planograma
    :param image: Imagen del planograma
    :return: Productos del planograma
    """
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
            folderName = "imagenActual/recortes/" + str(labels[product]) + "/"
            lastAdded = getLastAdded(folderName) + 1
            # Guarda la imagen recortada en una carpeta
            imagen_recortada.save(folderName + str(lastAdded) + ".jpg")

            # Agregar el producto a la fila
            rowProducts.append(product)
        # Agregar la fila a los productos
        products.append(rowProducts)

    return products


def scaleRectangles(rectangles, realSize, actualSize):
    """
    Función que ajusta los rectangulos con el tamaño de la imagen
    :param rectangles: Rectangulos a ajustar
    :param realSize: Tamaño real de la imagen
    :param actualSize: Tamaño de la imagen
    :return: Rectangulos ajustados
    """
    # Ajustar los rectangulos con el tamaño de la imagen real
    for rectangle in rectangles:
        rectangle["x"] = int(rectangle["x"] * realSize["width"] / actualSize["width"])
        rectangle["y"] = int(rectangle["y"] * realSize["height"] / actualSize["height"])
        rectangle["width"] = int(
            rectangle["width"] * realSize["width"] / actualSize["width"]
        )
        rectangle["height"] = int(
            rectangle["height"] * realSize["height"] / actualSize["height"]
        )

    return rectangles


@servidorWeb.route("/compareImages", methods=["POST"])
def compare():
    """
    Función que compara la imagen del planograma con la imagen actual
    :return: Resultado de la comparación
    """
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
            resultMatrix.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "currentProduct": product,
                    "expectedProduct": photoMatrix[row_index][column_index],
                    "isCorrect": is_correct,
                }
            )
            column_index += 1
        row_index += 1

    return jsonify({"resultMatrix": resultMatrix})


@servidorWeb.route("/classifyImage", methods=["POST"])
def classify():
    """
    Función que clasifica la imagen actual
    :return: Clasificación de la imagen actual
    """
    if "coordenadas" not in request.json["data"]:
        return "No image part in the form"

    rectangles = request.json["data"]["coordenadas"]["coordenadas"]

    # Obtener la imagen actual
    image = Image.open("imagenActual/imagenActual.jpg")

    # Obtener el tamaño de la imagen
    realSize = {"width": image.size[0], "height": image.size[1]}

    if "actualSize" in request.json["data"]:
        actualSize = request.json["data"]["actualSize"]
        # Ajustar los rectangulos con el tamaño de la imagen
        rectangles = scaleRectangles(rectangles, realSize, actualSize)

    #  Obtener el esquema del planograma
    scheme = getPlanogramScheme(rectangles)
    planogram = getPlanogramProducts(scheme, image)

    return planogram


@servidorWeb.route("/uploadImage", methods=["POST"])
def upload():
    """
    Función que recibe la imagen del planograma
    :return: Mensaje de confirmación
    """
    base64_data = request.json["imagen"]
    image_data = base64.b64decode(base64_data)
    imagen = Image.open(io.BytesIO(image_data))

    # erase previous image if exists
    if os.path.exists("imagenActual/imagenActual.jpg"):
        os.remove("imagenActual/imagenActual.jpg")

    # wait for the image to be deleted
    while os.path.exists("imagenActual/imagenActual.jpg"):
        pass

    if request.json["transpose"]:
        imagen = imagen.transpose(Image.ROTATE_270)

    # Saves images
    imagen.save("imagenActual/imagenActual.jpg")

    # Wait for the image to be saved
    while not os.path.exists("imagenActual/imagenActual.jpg"):
        pass

    return {"message": "ok"}


@servidorWeb.route("/getImageSize", methods=["GET"])
def getImageSize():
    """
    Función que obtiene el tamaño de la imagen
    :return: Tamaño de la imagen
    """
    image = Image.open("imagenActual/imagenActual.jpg")
    width, height = image.size
    return jsonify({"width": width, "height": height})


if __name__ == "__main__":
    servidorWeb.run(debug=False, host="0.0.0.0", port="8083")
