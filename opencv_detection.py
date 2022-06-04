# Built on top of Adrian Rosebrock work from pyimagesearch.

from cv2 import threshold
from imutils import paths
import numpy as np
import argparse
import cv2
import json
import time

path = "data/temp"

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
    help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

results={
    "sharp": {
        "positive": 0,
        "negative": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "motblurred-real": {
        "positive": 0,
        "negative": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "defblurred-real": {
        "positive": 0,
        "negative": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "motblurred-synth": {
        "positive": 0,
        "negative": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    },
    "defblurred-synth": {
        "positive": 0,
        "negative": 0,
        "totalTimeMs": 0,
        "avgTimeMs": 0
    }
}

for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    startTime = time.time()
    # LoG
    fm = variance_of_laplacian(gray)
    # FFT
    # (fm, res) = detect_blur_fft(gray)

    isBlurry = False

    if fm < args["threshold"]:
        isBlurry = True

    imageBlurType = imagePath.split("-")[1].split(" ")[0].split(".")[0].strip()

    if "blurred" in imageBlurType:
        imageBlurType = "%s-%s" % (imageBlurType, "real" if "real" in imagePath else "synth")

        if isBlurry:
            results[imageBlurType]["positive"] = results[imageBlurType]["positive"] + 1
        else:
            results[imageBlurType]["negative"] = results[imageBlurType]["negative"] + 1
    else:
        if not isBlurry:
            results["sharp"]["positive"] = results["sharp"]["positive"] + 1
        else:
            results["sharp"]["negative"] = results["sharp"]["negative"] + 1
    
    deltaTime = (time.time() - startTime) * 1000
    results[imageBlurType]["totalTimeMs"] = results[imageBlurType]["totalTimeMs"] + deltaTime
    results[imageBlurType]["avgTimeMs"] = results[imageBlurType]["totalTimeMs"] / (
        results[imageBlurType]["positive"] + results[imageBlurType]["negative"]
    )

    # For debugging
    # print(imagePath, " - {}: {:.2f}".format("Blurry" if isBlurry else "Not Blurry", fm))

allpositive = results["sharp"]["positive"] + results["motblurred-real"]["positive"] + results["defblurred-real"]["positive"] + results["motblurred-synth"]["positive"] + results["defblurred-synth"]["positive"]
allFalse = results["sharp"]["negative"] + results["motblurred-real"]["negative"] + results["defblurred-real"]["negative"] + results["motblurred-synth"]["negative"] + results["defblurred-synth"]["negative"]

totalResults = {
    "accuracy": allpositive / (allpositive + allFalse),
    "recall": results["sharp"]["positive"] / (results["sharp"]["positive"] + results["sharp"]["negative"]),
    "types": results
}

print("\nTOTAL RESULTS (Threshold = %s)" % args["threshold"])
print(json.dumps(totalResults, sort_keys=False, indent=4))