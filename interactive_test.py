import cv2 as cv2
import numpy as np
from random import shuffle
from os import listdir, makedirs, path
from shutil import copyfile
import tensorflow as tf
import matplotlib.pyplot as plt

input_folder = "./dataset/interact/"

if not path.exists("./dataset/interact2"):
    makedirs("./dataset/interact2")

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def load_image_test(image_file):

  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   256, 256)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def generate_images(model, test_input, tar):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

print("Cargando modelo...")

# Load weights into the new model
model = tf.keras.models.load_model('./models/GGrassNet_model.h5')

green_bk = cv2.merge([np.zeros((256,256,1), dtype=np.uint8), 
                           np.ones((256,256,1), dtype=np.uint8) * 255,
                           np.zeros((256,256,1), dtype=np.uint8)
                           ])

brown_bk = cv2.merge([np.zeros((256,256,1), dtype=np.uint8), 
                           np.ones((256,256,1), dtype=np.uint8) * 102,
                           np.ones((256,256,1), dtype=np.uint8) * 204
                           ])

yellow_bk = cv2.merge([np.ones((256,256,1), dtype=np.uint8) * 102, 
                           np.ones((256,256,1), dtype=np.uint8) * 255,
                           np.ones((256,256,1), dtype=np.uint8) * 255
                           ])

shine_green_bk = cv2.merge([np.ones((256,256,1), dtype=np.uint8) * 204, 
                           np.ones((256,256,1), dtype=np.uint8) * 255,
                           np.ones((256,256,1), dtype=np.uint8) * 102
                           ])

for f in listdir(input_folder):

    print("Procesando..." + f)

    image = cv2.imread(input_folder + f)
    image = cv2.resize(image, (256, 256))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    print("Generando máscaras..." + f)

    # Apply green color mask
    green = cv2.inRange(image, ((90 * 255) / 360, 0, 0), ((140 * 255) / 360, 255,255))
    # Apply brown color mask
    brown = cv2.inRange(image, ((30 * 255) / 360, 0, 0), ((40 * 255) / 360, 255,255))
    # Apply yellow color mask
    yellow = cv2.inRange(image, ((40 * 255) / 360, 0, 0), ((60 * 255) / 360, 255,255))
     # Apply shine green color mask
    shine_green = cv2.inRange(image, ((70 * 255) / 360, 0, 0), ((80 * 255) / 360, 255,255))

    image_green = cv2.bitwise_and(green_bk, green_bk,mask=green)
    image_brown = cv2.bitwise_and(brown_bk, brown_bk,mask=brown)
    image_yellow = cv2.bitwise_and(yellow_bk, yellow_bk,mask=yellow)
    image_shine_green = cv2.bitwise_and(shine_green_bk, shine_green_bk,mask=shine_green)

    print("Apilando mascaras..." + f)

    generated = cv2.add(image_green, image_green)
    generated = cv2.add(image_green, image_brown)
    generated = cv2.add(generated, image_yellow)
    generated = cv2.add(generated, image_shine_green)

    sample = np.concatenate((image, generated),axis=1)

    cv2.imwrite("./dataset/interact2/"+ f, sample)

test_dataset = tf.data.Dataset.list_files('./dataset/interact2/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)

for inp, target in test_dataset.take(5):
    generate_images(model, inp, target)

