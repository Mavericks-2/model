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

def getClassification(image) -> None:
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
    labels = ['Cheetos Torciditos',
              'Chips Fuego',
              'Chips Jalapeño',
              'Hut Nuts',
              'Maruchan PolloConVegetales',
              'Nissin CamaronPicante',
              'Takis Fuego',
              'Tostitos']

    predicted_label = labels[predicted_class]
    return predicted_class
