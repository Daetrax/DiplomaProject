# import the necessary packages
import imutils
from sklearn.feature_extraction import image as imageLib
import numpy as np
import argparse
import time
import cv2
import itertools
import math
import os


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

            # yield (x, y, image[y:math.floor(windowSize[1]/2), y:math.floor(windowSize[1]), x:math.floor(windowSize[0]/2), x:math.floor(windowSize[0])])


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())


# load the image and define the window width and height
# image = cv2.imread(args["image"])

(winW, winH) = (29, 29)

# loop over the image pyramid

def slideWithPyramid(image, path):
    for resized in pyramid(image, scale=1.5):
            # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=1, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)


def slide(image, mask):
    counter = 0

    features = []
    labels = []
    for (x, y, window) in sliding_window(image, stepSize=8, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue


        # print(imageName + "_" + str(counter) + ".tif")
        centerW = winW / 2
        centerH = winH / 2

        # features[counter] = np.asarray(window.getdata(), dtype=np.float64).reshape((window.size[1], window.size[0]))
        window_string = ""

        # TODO: check if this is needed, even though it's ultrasound
        window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        # window_string = np.array2string(window.ravel())
        window_string = window.ravel().tolist()

        features.append(window_string)
        # label 1
        if np.any(mask[int(y + centerW), int(x + centerH)] != 0):
            labels.append([0, 1])
        else:
            labels.append([1, 0])



        counter += 1

        # cv2.imwrite(imageName + "_" + str(counter) + ".tif", window)

        # since we do not have a classifier, we'll just draw the window
        # clone = image.copy()
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Window", clone)
        #
        # maskClone = mask.copy()
        # cv2.rectangle(maskClone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Mask", maskClone)
        # cv2.waitKey(1)
        # time.sleep(0.0000000001)


    # convertToTxt("testFile.txt", features, labels)
    return features, labels

# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):

    print("Saving", filename )
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))

def convertToTxt(filename, ndarray, rawLabels):

    print("Converting", filename)
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(2, dtype=np.uint).astype(str)))
        # for row, label in ndarray, rawLabels:
        for feature, label in itertools.zip_longest(ndarray, rawLabels):
            # row_str = row.astype(str)
            row_str = ' '.join(str(e) for e in feature)
            # label_str = labels[row[-1]]
            # feature_str = ' '.join(row_str[:-1])
            feature_str = row_str
            label_str = ' '.join(str(e) for e in label)
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))

# def mySlide(image, mask, stepSize):
#     for y in range(0, image.shape[0], stepSize):
#         for x in range(0, image.shape[1], stepSize):
#             return image[]

def saveCsv(image):
    value = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))
    np.savetxt("img_pixels.csv", value, delimiter=',')
