import onnx
import onnxruntime
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

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
    productMatrix = []
    for r in labelsMatrix:
        row = []
        for label in r:
            row.append(labels[label])
        productMatrix.append(row)

    return productMatrix

def compareMatrix(actual, real):
    diferences = []
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            if actual[i][j] != real[i][j]:
                diferences.append([(i, j), (actual[i][j], real[i][j])])
    return diferences

def getLenMatrix(matrix):
    n = 0
    for row in matrix:
        n += len(row)
    return n


if __name__ == "__main__":
    labelsMatrix =[[29,29,29,29,19,29,14,14],[7,7,9,24,7,19,8,23],[6,26,29,29,29,29,14,29,29]]
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

    realMatrix = [['ChipsJalapeño', 'BitzAlmendrasConSal', 'MaruchanCarneDeRes',
                   'TakisOriginal', 'HutNuts', 'TakisFuego', 'Runners', 'Churrumais'], 
                   [
                    'CheetosFlaminHot', 'NestléCarnationLecheEvaporada', 'ChipsFuego', 'PopKarameladas',
                    'SabritasSal', 'NissinCamaronPicante', 'MaruchanPolloConVegetales', 'NissinCarneDeRes',
                   ],
                   [
                    'FritosLimonYSal', 'RufflesQueso', 'NestléLaLecheraOriginal', 'TajínDulce', 'SalsaBúfaloClásica',
                    'Rancheritos', 'DoritosNachos', 'BitzCacahuateHabanero', 'BitzCacahuateHabanero', 'BitzCacahuatesEnchilados'
                   ]]
    actualMatrix = getProductMatrix(labels, labelsMatrix)
    print(compareMatrix(actualMatrix, realMatrix))

    nProducts = getLenMatrix(labelsMatrix)
    print("Porcentaje de acierto: ", (nProducts - len(compareMatrix(actualMatrix, realMatrix))) / nProducts * 100, "% \n")
