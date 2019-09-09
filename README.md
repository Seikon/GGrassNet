# GGrassNet
Generated Realistic grass imaged based on plain color 


## Introducción
Este proyecto pretende generar imagenes reales de texturas de césped basandose únicamente en matrices de colores planos tales como verde, marrón, amarillo...

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/1.JPG)

Para ello, se ha recurrido al modelo de deep learning Pix2Pix: (https://arxiv.org/pdf/1611.07004.pdf), en el cual se describe un modelo basado en Gans (redes neuronales generativas adversarias), que mediante un generador y un discriminador, aprende a traducir los pixeles de una imagen de entrada a una de salida.

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/2.png)

## Dataset
El dataset de entrenamiento ha sido extradio de la plataforma de aprendizaje de deep learning kaggle:

https://www.kaggle.com/archfx/paddygrass-distinguisher

Dicho dataset contiene más 500 imagenes de césped, las cuales han sido tratadas con funciones de numpy y opencv. Dicho tratamiento consiste en aplicar una serie de máscaras de color personalizadas a la imagen, de tal forma que podemos codificar cada pixel de cada brizna de césped con su color correspondiente: (bitwise_and() para los amigos). Para facilitar la aplicación de máscaras de color, se ha echo una transformación previa del canal de color RBG al canal HSV, donde H representa la gama de colores en 360 slots diferentes. Como las imagenes constan de matrices de 8 bits, se aplicado un factor de conversión para transformar estos 360 slots a 255 posibles valores:
### Imagen original

### Máscara de color verde

### Máscara de color amarillo

### Máscara de color naranja y marrón

### Máscara de color verdes claros

Finalmente sumando todas estas máscaras obtenemos la imagen que codifica la información de entrada:
### Resultado suma de todas las matrices (256, 256):

## Entrenamiento
El módelo se ha entrenado durante 150 épocas. Se ha observado que para 20 o 30 épocas el modelo ya convergía, pero generaba pequeños artefactos que, siendo poco perceptibles, a veces se convertian en atleatorios:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/5.JPG)

El módelo ha sido entrenado con una targeta gráfica nvidia gtx 1060 3GB, lo que a limitado en parte el proceso y a obligado a reducir parametros del entrenamiento como el batch size. A continuación se muestran los parámetros utilizados para el entrenamiento:

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

Además se ha conservado el factor lambda de regularización por defecto que recomiendan los autores originales del modelo Pix2Pix a 100.

## Test y... pruebelo usted mismo!

## Agradecimientos
A Carlos Santana Vega (https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg) por sus incasable labor de divulgación científica que llega desde ingenieros hasta estudiantes de secundaria que dan sus primeros pasos en deep learning.

A Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros, autores originales de Pix2Pix por hacer público un conocimiento que en el futuro cercano estará muy presente.

A Victor Valderrama Arroyo (https://www.deviantart.com/victorvicius) por su apoyo al proyecto, su aportación de ideas y generación de datasets de manera totalmente desinteresada. Ah... y por aportar su hardware rtx 2050 8GB para hacer pruebas de concepto rápidas que a mi me costaban la vida.

Al usuario Aruna Jayasena (https://www.kaggle.com/archfx) por su aportación de manera pública al dataset del cual se ha nutrido este proyecto

A ti, lector interesado en la IA y en el machine learning por tu tiempo de lectura.





