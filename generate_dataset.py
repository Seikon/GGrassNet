import cv2 as cv2
import numpy as np
from random import shuffle
from os import listdir, makedirs, path
from shutil import copyfile

def suffle(train_output_folder, cross_output_folder, test_output_folder, src_folder):
    
    mode = 0 # 0 -> Train, 1 -> Cross, 2 -> Test

    num_files = len(listdir(src_folder)) - 1

    num_train = round(num_files * 0.7)
    num_cross = round(num_files * 0.15)
    num_val = round(num_files * 0.15)

    images = [[f] for f in listdir(src_folder)]

    shuffle(images)

    train = images[:num_train]
    cross = images[num_train:num_cross]
    test = images[num_train + num_cross: num_files - 1]

    counter = 1

    print("-- Train --")
    for f in train:
        print(f[0])
        copyfile(src_folder + "/" + f[0], train_output_folder + "/" + str(counter) + ".jpg")
        counter += 1

    counter = 1

    print("-- Cross --")
    for f in cross:
        print(f[0])
        copyfile(src_folder + "/" + f[0], cross_output_folder + "/" + str(counter) + ".jpg")
        counter += 1

    counter = 1

    print("-- Test --")
    for f in test:
        print(f[0])
        copyfile(src_folder + "/" + f[0], test_output_folder + "/" + str(counter) + ".jpg")
        counter += 1

print("Creating directories...")

if not path.exists("./dataset/train"):
    makedirs("./dataset/train")

if not path.exists("./dataset/val"):
    makedirs("./dataset/val")

if not path.exists("./dataset/test"):
    makedirs("./dataset/test")

if not path.exists("./dataset/merge"):
    makedirs("./dataset/merge")

if not path.exists("./models"):
    makedirs("./models")

input_folder = "./dataset/input/"

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

    print("Processing..." + f)

    image = cv2.imread(input_folder + f)
    image = cv2.resize(image, (256, 256))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

    generated = cv2.add(image_green, image_brown)
    generated = cv2.add(generated, image_yellow)
    generated = cv2.add(generated, image_shine_green)

    sample = np.concatenate((image, generated),axis=1)

    cv2.imwrite("./dataset/merge/" + f, sample)


suffle(
            "./dataset/train", 
            "./dataset/val", 
            "./dataset/test",
            "./dataset/merge")