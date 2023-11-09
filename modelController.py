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

if __name__ == "__main__":
    labelsMatrix = [[24, 9, 29, 14, 33, 12, 14], [8, 19, 27, 15, 19, 26, 10], [19, 19, 19, 23, 14, 29, 29, 19, 21, 19, 12]]
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
    print(getProductMatrix(labels, labelsMatrix))