
import os
import random
import cv2
import numpy as np
from sklearn.cluster import k_means

dataPath="data/Totaldump"
allFiles = os.listdir(dataPath)
sharpFiles = [file for file in allFiles if "sharp" in file]

def syntheticallyBlurFiles(blurType = "defblurred"):
    for count, fileName in enumerate(sharpFiles):
        person = fileName.split("-")[0].strip()        
        kernelSize = random.randint(12, 75)

        newFileName = "%s/%s-%s-%s (%s_k%s).jpeg" % (dataPath, person, blurType, "synth", count, kernelSize)
        
        originalImagePath = "%s/%s" % (dataPath, fileName)
        originalImage = cv2.imread(originalImagePath)
        

        if blurType == "motblurred":
            imageBlurred = getMotionBlurredImage(originalImage, kernelSize)
        else:
            imageBlurred = cv2.blur(src=originalImage, ksize=(kernelSize, kernelSize))

        # cv2.imshow("%s k=%s" % (originalImagePath, kernelSize), imageBlurred)
        # cv2.imwrite(newFileName, imageBlurred)
        print(newFileName)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


def getMotionBlurredImage(image, ksize):
    kernel_h = np.zeros((ksize, ksize))
    kernel_h[int((ksize - 1)/2), :] = np.ones(ksize)
    kernel_h /= ksize
    horizonal_mb = cv2.filter2D(image, -1, kernel_h)

    return horizonal_mb

# img = cv2.imread(hello)
# avging = cv2.blur(img,(20,20))
   
# cv2.imshow('Original',img)
# cv2.imshow('Averaging',avging)

# cv2.waitKey(0)


# print(allFiles)
# print("\n------\n")
# print(sharpFiles)
syntheticallyBlurFiles("defblurred")