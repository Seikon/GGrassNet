# GGrassNet
Generación de imagenes realistas de céspedes basado en colores planos


## Introducción
Este proyecto pretende generar imagenes reales de texturas de césped basandose únicamente en matrices de colores planos tales como verde, marrón, amarillo...

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/1.JPG)

Para ello, se ha recurrido al modelo de deep learning Pix2Pix: (https://arxiv.org/pdf/1611.07004.pdf), en el cual se describe un modelo basado en Gans (redes neuronales generativas adversarias), que mediante un generador y un discriminador, aprende a traducir los pixeles de una imagen de entrada a una de salida.

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/2.png)

## Caso Práctico. Regenerar un terreno desertizado a través de GGrassNet
Imaginemos que partimos de la imagen de un terreno desertizado como el de la siguiente imagen:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/12.jpg)

Através de GGrassNet es posible generar una imagen del mismo terreno regenerado con césped.

Como queremos generar una imagen del terreno pero cubierto de césped, hemos de alguna forma transformar la imagen de entrada a una que nuestro modelo pueda entender. No vale cualquier imagen de colores verdes, marrones o amarrillos, ya que no estariamos respetando la estructura de la imagen original y por tanto debemos de alguna forma preservar la estructura básica de la red.

Para ello se describe el siguiente procedimiento:

### 1. Extracción de características
Una forma de extraer información de la imagen es a través de sus bordes o de contornos:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/13.jpg)

### 2. Generación de ROIS (Regions of interest)
Como lo que nos interesa modificar de la imagen no son los bordes, sino las regiones que contiene dentro de ellos, hemos de resaltar dichas regiones invirtiendo el color de la imagen (operación NOT sobre matrices):

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/14.jpg)

### 2. Generación de RPN (Random ponderated noise)
Las regiones de interes deben ser llenadas con los colores seleccionados con los que entrenamos a GGrassNet, por ello hemos de llenar dichas regiones de alguna manera. Como los pixeles de las imagenes de césped no son completamente atleatorias (el verde por ejemplo está mas presente que otros colores), el ruido atleatorio que hemos de generar dentro de las ROIS debe ser ponderado, es decir, que ciertos colores aparezcan con mas frecuencia que otros. A continuación se puede ver una imagen del algoritmo en proceso de generación:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/gifRPN.gif)

### 3. Fusión de imagen RPN + ROI (Random ponderated noise)
A continuación llenamos nuestras ROIS con la máscara RPN generada anteriormente y obtenemos la imagen de ROIS llena completamente con los colores planos que GGrassNet entenderá:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/15.JPG)

### 4. FastForward através de GrassNet:
Finalmente, pasamos la imagen a través de la red obteniendo los siguientes resultados:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/16.jpg)

Y así conseguimos generar una imagen restaurada con césped a tarevés de dicha imagen de un suelo desertificado:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/17.jpg)

## Dataset
El dataset de entrenamiento ha sido extradio de la plataforma de aprendizaje de deep learning kaggle:

https://www.kaggle.com/archfx/paddygrass-distinguisher

Dicho dataset contiene más 500 imagenes de césped, las cuales han sido tratadas con funciones de numpy y opencv. Dicho tratamiento consiste en aplicar una serie de máscaras de color personalizadas a la imagen, de tal forma que podemos codificar cada pixel de cada brizna de césped con su color correspondiente: (bitwise_and() para los amigos). Para facilitar la aplicación de máscaras de color, se ha echo una transformación previa del canal de color RBG al canal HSV, donde H representa la gama de colores en 360 slots diferentes. Como las imagenes constan de matrices de 8 bits, se aplicado un factor de conversión para transformar estos 360 slots a 255 posibles valores:

### Imagen original
![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/4.JPG)

### Máscaras de colores (verde (V), amarillo (A), verdes_claros (VA), marrones (M))
![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/3.JPG)

Finalmente sumando todas estas máscaras obtenemos la imagen que codifica la información de entrada:
### Resultado suma de todas las matrices (256, 256) (G = V + A + VA + M):

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/7.JPG)

## Entrenamiento
El módelo se ha entrenado durante 150 épocas. Se ha observado que para 20 o 30 épocas el modelo ya convergía, pero generaba pequeños artefactos que, siendo poco perceptibles, a veces se convertian en atleatorios:

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/5.JPG)

El módelo ha sido entrenado con una targeta gráfica nvidia gtx 1060 3GB, lo que a limitado en parte el proceso y a obligado a reducir parametros del entrenamiento como el batch size. A continuación se muestran los parámetros utilizados para el entrenamiento:

    BUFFER_SIZE = 400
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    
Se ha ejecutado también data augmentation con el fin de aumentar el set de datos. Las tecnicas empleadas han sido flip y random cropping.

Además se ha conservado el factor lambda de regularización por defecto que recomiendan los autores originales del modelo Pix2Pix a 100.

## Test y... pruébelo usted mismo!

A continuación se muestra una serie de test generados tras el entrenamiento de 150 épocas:
![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/8.JPG)
![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/9.JPG)
![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/10.JPG)

#### Para una prueba interactiva, descarga este repositorio:

    git clone https://github.com/Seikon/GGrassNet.git

#### Instala las librerias necesarias:

    pip install tensorflow-gpu==2.0.0-rc0 (o la versión cpu en caso de no disponer de gpu)
    pip install numpy==1.17.2
    pip install matplotlib
    pip install opencv-python
    pip install IPython
    pip install future

#### Entrenamiento desde 0

Para descargar el dataset completo puedes usar este enlace (y descomprimelo en la carpeta raiz del proyecto):
            https://drive.google.com/file/d/1jp8lC0aa_YPDRkfx2-e5vFiPgCaR-gnA/view

Primeramente ejecuta en la consola de comandos el archivo que genera el dataset:

    python generate_dataset.py

Luego entrena el modelo durante las épocas que elijas: (dentro del código, en la linea 281, modifica la variable EPOCHS, para modificar el número de épocas de entrenamiento):

    python Pix2Pix.py

*Este proceso tardará mas o menos dependiendo de la GPU / CPU que tengas

Una vez entrenado el modelo, inserta en la carpeta "interact" las images que quieras para testear el modelo con sus propias imágenes. He incluido unas de ejemplo sacadas directamente de google.

Con el siguiente comando, el modelo recorrerá y ejecutará un fast forward a través de tus imagenes:

    python interactive_test.py

#### Usando modelo pre-entrenado

Para descargar el modelo pre-entrenado puedes usar este enlace (y descomprimelo en la carpeta raiz del proyecto):
            https://drive.google.com/file/d/1D-bBZ-3emM5M-ERX2CbmComN6QZhjDxP/view?usp=sharing

Con el siguiente comando, el modelo recorrerá y ejecutará un fast forward a través de tus imagenes usando el modelo preentrenado de la carpeta models:

    python interactive_test.py

## Agradecimientos
A Carlos Santana Vega (https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg) por sus incasable labor de divulgación científica que llega desde ingenieros hasta estudiantes de secundaria que dan sus primeros pasos en deep learning.

A Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros, autores originales de Pix2Pix por hacer público un conocimiento que en el futuro cercano estará muy presente.

A Victor Valderrama Arroyo (https://www.deviantart.com/victorvicius) por su apoyo al proyecto, su aportación de ideas y generación de datasets de manera totalmente desinteresada. Ah... y por aportar su hardware rtx 2050 8GB para hacer pruebas de concepto rápidas que a mi me costaban la vida.

Al usuario Aruna Jayasena (https://www.kaggle.com/archfx) por su aportación de manera pública al dataset del cual se ha nutrido este proyecto

A ti, lector interesado en la IA y en el machine learning por tu tiempo de lectura :)

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/11.JPG)
