import cv2
import numpy as np
import random

def generateImageRPS(width, height):

    tar = np.zeros((width, height, 3), dtype=np.uint8)

    # Contains all posible ponderized values
    # 1 -> 40 % Green values
    # 2 -> 20 % Brown values
    # 3 -> 10 % yellow values
    # 4 -> 10 % shine green values
    # 5 -> 20 % black values
    ponderizedBagOptions = [1,1,1,1,2,2,3,4,5,5]

    for indRow in range(height):

        for indCol in range(width):

            randPos = random.randint(0, 9)

            choose = ponderizedBagOptions[randPos]

            if choose == 1:
                tar[indRow, indCol] = [0, 255, 0]
            elif choose == 2:
                tar[indRow, indCol] = [0, 102, 204]
            elif choose == 3:
                tar[indRow, indCol] = [102, 255, 255]
            elif choose == 4: 
                tar[indRow, indCol] = [204, 255, 102]
            else:
                tar[indRow, indCol] = [0, 0, 0]

            cv2.imshow("tar", tar)
            cv2.waitKey(1)

    return tar
        

image = cv2.imread("test.jpg")

image = cv2.resize(image, (256,256))

contoured = cv2.Canny(image, 100, 500)

# Invert mask in order to apply it only in not border areas
# This way, we can terrize the floor without losing information
contoured = cv2.bitwise_not(contoured)

cv2.imshow("contoured_n", contoured)

# Generate Random ponderated noise
RPSNoise = generateImageRPS(256,256)

# Meld maks with RPS Noise
masked = cv2.bitwise_and(RPSNoise, RPSNoise, mask=contoured)

cv2.imshow("masked", masked)

cv2.imwrite("imageRPS", masked)

cv2.waitKey(0)

    