# 🍌 BananaPro Classifier

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)

Sistema avanzado de clasificación de bananos que combina visión por computadora y aprendizaje profundo para identificación en tiempo real con autenticación facial.

<img width="1319" height="610" alt="image" src="https://github.com/user-attachments/assets/e9cc836a-494f-474e-90c2-de1ddec4f012" />


## ✨ Características Principales

- **Clasificación en Tiempo Real** de bananos (maduros, verdes, otros)
- **Autenticación Facial** para acceso seguro al sistema
- **Detección Multi-modal**:
  - Reconocimiento por color (HSV)
  - Detección de movimiento
  - Modelo Keras para clasificación precisa
- **Interfaz Premium** con modo oscuro y diseño elegante
- **Registro Automático** de eventos en CSV
- **Sistema de Snapshots** para almacenar imágenes clasificadas

## 🛠️ Tecnologías Utilizadas

| Tecnología       | Uso en el Proyecto                     |
|------------------|---------------------------------------|
| TensorFlow/Keras | Modelo de clasificación de imágenes   |
| OpenCV           | Procesamiento de imágenes/video       |
| Streamlit        | Interfaz web interactiva              |
| Streamlit-WebRTC | Transmisión de video en tiempo real   |
| Haarcascades     | Detección facial                      |
| Pandas           | Registro y manejo de datos            |

## 🚀 Cómo Empezar

### Prerrequisitos

- Python 3.8+
- Git
- Pip

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TRAUSAURIO/clasificador-de-banano.git
   cd clasificador-de-banano
