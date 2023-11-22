"""
@authors: Pablo González, José Ángel García, Erika Marlene
@description: Controlador del modelo de clasificación el cual
interactúa con el servidor para la clasificación de los productos.
"""

import onnx
import onnxruntime
import numpy as np
from torchvision import transforms

# Lectura del modelo
onnx_model_path = "modelo.onnx"
onnx_model = onnx.load(onnx_model_path)

# Sesión de ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_model_path)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


def getClassification(image):
    """
    Función que clasifica una imagen de un producto
    :param image: Imagen del producto
    :return: Clasificación del producto
    """
    # Transformar la imagen
    image = transform(image)
    # Convertir la imagen transformada a un tensor
    image_tensor = image.unsqueeze(0)

    # Ejecutar la inferencia en el modelo ONNX
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    ort_inputs = {input_name: image_tensor.cpu().numpy()}
    ort_outs = ort_session.run([output_name], ort_inputs)

    # Las predicciones están en ort_outs[0], que es un ndarray de NumPy
    predictions = ort_outs[0]
    predicted_class = np.argmax(predictions)

    return predicted_class


def getProductMatrix(labels, labelsMatrix):
    """
    Función que obtiene la matriz de productos
    :param labels: Etiquetas de los productos
    :param labelsMatrix: Matriz de etiquetas
    :return: Matriz de productos
    """
    productMatrix = []
    for r in labelsMatrix:
        row = []
        for label in r:
            row.append(labels[label])
        productMatrix.append(row)

    return productMatrix

def compareMatrix(actual, real):
    """
    Función que compara dos matrices
    :param actual: Matriz actual
    :param real: Matriz real
    :return: Diferencias entre las matrices
    """
    diferences = []
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            if actual[i][j] != real[i][j]:
                diferences.append([(i, j), (actual[i][j], real[i][j])])
    return diferences

def getLenMatrix(matrix):
    """
    Función que obtiene el número de elementos de una matriz
    :param matrix: Matriz
    :return: Número de elementos de la matriz
    """
    n = 0
    for row in matrix:
        n += len(row)
    return n


if __name__ == "__main__":
    labelsMatrix =[[24,9,34,25,7,12,14],[8,7,7,7,7,26,10],[7,7,19,7,7,29,29,7,21,29,7]]
    labels = ['BitzAlmendrasConSal',
              'BitzCacahuateEnchilado',
              'BitzCacahuateHabanero',
              'BitzCacahuatesEnchilados',
              'BotaneraChilito',
              'CacahuatesJaponésLeo',
              'CacahuatesSalBokados',
              'CheetosFlaminHot',
              'CheetosTorciditos',
              'ChipsFuego',
              'ChipsJalapeño',
              'ChipsPapatinas',
              'Churrumais',
              'DelPrimoSalsaGuacamole',
              'DoritosNachos',
              'FritosLimonYSal',
              'HutNuts',
              'LeoMixBotanero',
              'MaruchanCarneDeRes',
              'MaruchanPolloConVegetales',
              'NestléCarnationLecheEvaporada',
              'NestléLaLecheraOriginal',
              'NissinCamaronPicante',
              'NissinCarneDeRes',
              'PopKarameladas',
              'Rancheritos',
              'RufflesQueso',
              'Runners',
              'SabritasSal',
              'SalsaBúfaloClásica',
              'SemillasDeCalabazaBokados',
              'SemillasDeGirasol',
              'TajínDulce',
              'TakisFuego',
              'TakisOriginal',
              'Tostitos']

    realMatrix_planogramEx_1_2_3 = [
                    ['ChipsJalapeño', 'BitzAlmendrasConSal', 'MaruchanCarneDeRes',
                   'TakisOriginal', 'HutNuts', 'TakisFuego', 'Runners', 'Churrumais'], 
                   [
                    'CheetosFlaminHot', 'NestléCarnationLecheEvaporada', 'ChipsFuego', 'PopKarameladas',
                    'SabritasSal', 'NissinCamaronPicante', 'MaruchanPolloConVegetales', 'NissinCarneDeRes',
                   ],
                   [
                    'FritosLimonYSal', 'RufflesQueso', 'NestléLaLecheraOriginal', 'TajínDulce', 'SalsaBúfaloClásica',
                    'Rancheritos', 'DoritosNachos', 'BitzCacahuateHabanero', 'BitzCacahuatesEnchilados'
                   ]]
    realMatrix_planogramEx_4_5_6_7 = [['PopKarameladas', 'ChipsFuego', 'TakisOriginal', 'Rancheritos', 'TakisFuego', 'Churrumais', 'DoritosNachos'], 
                                     ['CheetosFlaminHot', 'RufflesQueso', 'FritosLimonYSal', 'NestléCarnationLecheEvaporada', 'HutNuts', 'SabritasSal', 'MaruchanPolloConVegetales', 'NissinCarneDeRes'], 
                                     ['BitzCacahuatesEnchilados', 'BitzCacahuateHabanero', 'NestléLaLecheraOriginal', 'TajínDulce', 'SalsaBúfaloClásica']]

    realMatrix_planogramEx_9 = [['PopKarameladas', 'ChipsFuego', 'TakisOriginal', 'Rancheritos', 'TakisFuego', 'Churrumais', 'DoritosNachos'],
                                ['HutNuts', 'CheetosFlaminHot', 'Runners', 'FritosLimonYSal', 'NestléCarnationLecheEvaporada', 'MaruchanPolloConVegetales', 'NissinCamaronPicante', 'NissinCarneDeRes'],
                                ['MaruchanCarneDeRes', 'ChipsJalapeño', 'SabritasSal', 'RufflesQueso', 'BitzAlmendrasConSal', 'SalsaBúfaloClásica', 'TajínDulce', 'NestléLaLecheraOriginal', 'BitzCacahuatesEnchilados', 'BitzCacahuateHabanero']]
    
    realMatrix_planogramEx_10__20 = [['PopKarameladas', 'ChipsFuego', 'TakisOriginal', 'Rancheritos', 'TakisFuego', 'Churrumais', 'DoritosNachos'],
                                     ['HutNuts', 'CheetosFlaminHot', 'Runners', 'FritosLimonYSal', 'SabritasSal', 'RufflesQueso', 'ChipsJalapeño'],
                                     ['MaruchanPolloConVegetales', 'MaruchanCarneDeRes', 'NissinCamaronPicante', 'NissinCarneDeRes', 'NestléCarnationLecheEvaporada', 'BitzAlmendrasConSal', 'SalsaBúfaloClásica', 'TajínDulce', 'NestléLaLecheraOriginal', 'BitzCacahuatesEnchilados', 'BitzCacahuateHabanero'],
                                     ]

    actualMatrix = getProductMatrix(labels, labelsMatrix)
    matrizDiferencia = compareMatrix(actualMatrix, realMatrix_planogramEx_10__20)
    print(matrizDiferencia)

    nProducts = getLenMatrix(labelsMatrix)
    percentage = (nProducts - len(matrizDiferencia)) / nProducts * 100
    print("Porcentaje de acierto: ", percentage, "% \n")
