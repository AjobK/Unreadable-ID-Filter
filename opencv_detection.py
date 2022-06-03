# import the necessary packages
from imutils import paths
import argparse
import cv2
import json

path = "data/temp"

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
    help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

results={
    "sharp": {
        "correct": 0,
        "incorrect": 0
    },
    "motblurred": {
        "correct": 0,
        "incorrect": 0
    },
    "defblurred": {
        "correct": 0,
        "incorrect": 0
    }
}

# loop over the input images
for imagePath in paths.list_images(args["images"]):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    isBlurry = False

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < args["threshold"]:
        isBlurry = True

    text = "Blurry" if isBlurry else "Not Blurry"

    # show the image
    # cv2.rectangle(image, (0, 0), (500, 100), (0, 0, 0), -1)
    # cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # scale_percent = 50 # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)

    imageBlurType = imagePath.split("-")[1].split(" ")[0].strip()
    if imageBlurType in ["motblurred", "defblurred"]:
        if isBlurry:
            results[imageBlurType]["correct"] = results[imageBlurType]["correct"] + 1
        else:
            results[imageBlurType]["incorrect"] = results[imageBlurType]["incorrect"] + 1
    else:
        if not isBlurry:
            results["sharp"]["correct"] = results["sharp"]["correct"] + 1
        else:
            results["sharp"]["incorrect"] = results["sharp"]["incorrect"] + 1

    print(imagePath, " - {}: {:.2f}".format(text, fm))

    # cv2.imshow("Image", cv2.resize(image, dim, interpolation = cv2.INTER_AREA))
    # key = cv2.waitKey(0)

print("\nTOTAL RESULT")
print(json.dumps(results, sort_keys=False, indent=4))