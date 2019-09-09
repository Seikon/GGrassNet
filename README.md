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

![alt text](https://raw.githubusercontent.com/Seikon/GGrassNet/master/docu/5.jpg)



