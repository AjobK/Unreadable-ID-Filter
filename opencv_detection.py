# import the necessary packages
from imutils import paths
import numpy as np
# from pyimagesearch.blur_detector import detect_blur_fft
import argparse
import cv2
import json

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

for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # LoG
    # fm = variance_of_laplacian(gray)
    # FFT
    (fm, res) = detect_blur_fft(gray)

    isBlurry = False

    if fm < args["threshold"]:
        isBlurry = True

    text = "Blurry" if isBlurry else "Not Blurry"

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

    # For debugging
    print(imagePath, " - {}: {:.2f}".format(text, fm))

print("\nTOTAL RESULT")
print(json.dumps(results, sort_keys=False, indent=4))