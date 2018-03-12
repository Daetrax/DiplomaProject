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