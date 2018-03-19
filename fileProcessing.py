import os
import cv2

def filterImagesWithNoMask(directory, destination):

    result = []
    for name in os.listdir(directory):

        filename = directory + name
        img = cv2.imread(filename)
        mask = cv2.imread(filename.replace(".tif", "") + "_mask.tif", cv2.IMREAD_GRAYSCALE)


        if cv2.countNonZero(mask) != 0:
            result.append(name)
            cv2.imwrite(destination + name, img)
            cv2.imwrite(destination + name.replace(".tif", "") + "_mask.tif", mask)

    return result


def separateImageAndMask(directory, destination):
    "This separates patient images into separate folders and removes masks. A preprocess step before making motion vectors"

    for name in os.listdir(directory):
        if "mask" in name:
            continue

        filename = directory + name

        patientDirectory = destination + name.split("_")[0] + "\\"

        if not os.path.exists(patientDirectory):
            os.makedirs(patientDirectory)

        image = cv2.imread(directory + name)

        cv2.imwrite(patientDirectory + name, image)

