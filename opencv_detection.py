# Built on top of Adrian Rosebrock work from pyimagesearch.

from cv2 import threshold
from skimage.filters import edges
from skimage import io, color
from imutils import paths
import numpy as np
import argparse
import cv2
import json
import time

path = "data/temp"

def variance_of_laplacian(image):
    theVar = cv2.Laplacian(image, cv2.CV_64F).var()
    return theVar

def detect_blur_fft(image, size=60, thresh=10):
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)

	return (mean, mean <= thresh)

def sci_variance_of_laplacian(image):
    theVar = edges.laplace(image, 3).var()
    return theVar

def sci_detect_blur_fft(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def normalizeData(data):
    newData = (data - np.min(data)) / (np.max(data) - np.min(data)) * 100
    print(newData)
    return newData

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
    help="focus measures that fall below this value will be considered 'blurry'")
ap.add_argument("-c", "--combination", default="ocv-log",
    help="The framework and algorithm combination used")
args = vars(ap.parse_args())

results={
    "sharp": {
        "true": 0,
        "false": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "motblurred-real": {
        "true": 0,
        "false": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "motblurred-synth": {
        "true": 0,
        "false": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "defblurred-real": {
        "true": 0,
        "false": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "defblurred-synth": {
        "true": 0,
        "false": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    }
}

lowestFm = [9999, "none"]
highestFm = [-9999, "none"]

for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = normalizeData(gray)

    startTime = time.time()

    fm = variance_of_laplacian(gray)

    if args["combination"].startswith("ocv-"):
        if args["combination"].endswith("log"):
            fm = variance_of_laplacian(gray)
        elif (args["combination"].endswith("fft")):
            (fm, res) = detect_blur_fft(gray)

    elif args["combination"].startswith("sci-"):
        if args["combination"].endswith("log"):
            fm = sci_variance_of_laplacian(gray)
        elif (args["combination"].endswith("fft")):
            (fm, res) = sci_detect_blur_fft(gray)

    isBlurry = False

    if fm < lowestFm[0]:
        lowestFm = [fm, imagePath]
    elif fm > highestFm[0]:
        highestFm = [fm, imagePath]


    if fm < args["threshold"]:
        isBlurry = True

    imageBlurType = imagePath.split("-")[1].split(" ")[0].split(".")[0].strip()

    if "blurred" in imageBlurType:
        imageBlurType = "%s-%s" % (imageBlurType, "real" if "real" in imagePath else "synth")

        if isBlurry:
            results[imageBlurType]["true"] = results[imageBlurType]["true"] + 1
        else:
            results[imageBlurType]["false"] = results[imageBlurType]["false"] + 1
    else:
        if not isBlurry:
            results["sharp"]["true"] = results["sharp"]["true"] + 1
        else:
            results["sharp"]["false"] = results["sharp"]["false"] + 1
    
    deltaTime = (time.time() - startTime) * 1000
    results[imageBlurType]["totalTimeMs"] = results[imageBlurType]["totalTimeMs"] + deltaTime
    results[imageBlurType]["avgTimeMs"] = round(results[imageBlurType]["totalTimeMs"] / (
        results[imageBlurType]["true"] + results[imageBlurType]["false"]
    ), 2)

    # For debugging
    # print(imagePath, " - {}: {:.2f}".format("Blurry" if isBlurry else "Not Blurry", fm))

truePositives = results["sharp"]["true"]
trueNegatives = results["motblurred-real"]["true"] + results["defblurred-real"]["true"] + results["motblurred-synth"]["true"] + results["defblurred-synth"]["true"]

falsePositives = results["sharp"]["false"]
falseNegatives = results["motblurred-real"]["false"] + results["defblurred-real"]["false"] + results["motblurred-synth"]["false"] + results["defblurred-synth"]["false"]

allPositive = trueNegatives + truePositives
allFalse = falsePositives + falseNegatives

totalResults = {
    "combo": args["combination"],
    "accuracy": round(allPositive / (allPositive + allFalse) * 100, 2),
    "recall": round(truePositives / (truePositives + falsePositives) * 100, 2),
    "truthMatrix": [truePositives, trueNegatives, falsePositives, falseNegatives],
    "types": results
}

print("\nTOTAL RESULTS (Threshold = %s)" % args["threshold"])
print(json.dumps(totalResults, sort_keys=False, indent=4))

print(lowestFm)
print(highestFm)