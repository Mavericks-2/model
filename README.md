# ShelfMate Backend - Flask

## Equipo 1 - CEM

Integrantes:

- A01751655 Cortés Olvera Gabriela
- A01745865 García Gómez José Ángel
- A01745096 González de la Parra Pablo
- A01751580 Islas Montiel Zaide
- A01745371 Sánchez Bahnsen Elisa
- A01382889 Ana Martínez Barbosa
- A01706870 José María Ibarra Pérez
- A01745158 García Sánchez Erika Marlene

## Tecnologías utilizadas

- [Python](https://www.python.org/) - v3.9.7
- [Flask](https://flask.palletsprojects.com/en/2.0.x/) - v2.0.2

## Requisitos del sistema

- [Git](https://git-scm.com/)
- [Python](https://www.python.org/) - v3.9.7

## Instalación

1. Clonar el repositorio

```bash
git clone git@github.com:Mavericks-2/model.git
```

2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

3. Correr el proyecto

```bash
python servidor.py
```

4. Abrir el navegador en la dirección [http://localhost:8083](http://localhost:8083)

## Documentación de rutas

### POST /compareImages

- **Body:**
  ```json
  {
      "data": {
          "planogram": {
              "coordenadas": [
                  [{"x": 0, "y": 0, "width": 0, "height": 0}, ...],
                  // ... Additional rows
              ]
          },
          "actualPlanogram": {
              "coordenadas": [
                  [{"x": 0, "y": 0, "width": 0, "height": 0}, ...],
                  // ... Additional rows
              ]
          }
      }
  }
  ```
- **Response:**
  ```json
  {
    "resultMatrix": [
      {
        "row": 0,
        "column": 0,
        "currentProduct": 0,
        "expectedProduct": 0,
        "isCorrect": true
      }
      // ... Additional entries
    ]
  }
  ```

### POST /classifyImage

- **Body:**
  ```json
  {
    "data": {
      "coordenadas": {
        "coordenadas": [
          { "x": 0, "y": 0, "width": 0, "height": 0 }
          // ... Additional rectangles
        ]
      },
      "actualSize": {
        "width": 0,
        "height": 0
      }
    }
  }
  ```
- **Response:**
  ```json
  [
    [0, 1, 2]
    // ... Additional rows
  ]
  ```

### POST /uploadImage

- **Body:**
  ```json
  {
    "imagen": "base64_string",
    "transpose": false
  }
  ```
- **Response:**
  ```json
  {
    "message": "ok"
  }
  ```

### GET /getImageSize

- **Response:**
  ```json
  {
    "width": 0,
    "height": 0
  }
  ```

## Licencia

[MIT](https://choosealicense.com/licenses/mit/)

La licencia de este proyecto es MIT, por lo que puede ser utilizado por cualquier persona, siempre y cuando se le dé el crédito correspondiente a los autores originales.
