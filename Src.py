# import the necessary packages
import os
import cv2

from NN import LogisticRegression as lr
import numpy as np

desktopDirectory = "D:/DP/Data/train/train/"
notebookDirectory = "C:/DP/Data/train/"

directory = notebookDirectory
dest = "D:/DP/Data/Preprocess_1/train/"
# len = len(os.listdir(directory))


# deseralizer = MyDeserializer.MyDeserializer(directory=dataDirectory, streams=[dict(name='Features', shape=(361,)), dict(name='Labels', shape=(2,))])
#
# deseralizer.createDictionary(dataDirectory)
#
# deseralizer.get_chunk(2)


# Define the data dimensions
input_dim = 841
num_output_classes = 2

import ComputerVisionAssignments.motionAndVideo as motion
# motion.detectBasicMotion("D:\\DP\\Data\\Preprocess\\motion\\1\\1.avi", 350)
# motion.denseOpticalFlow("D:\\DP\\Data\\Preprocess\\motion\\1\\1.avi")
# motion.lucasKanadeOpticalFlow("D:\\DP\\Data\\Preprocess\\motion\\1\\1.avi")

def createVideosFromImages():
    import fileProcessing, ComputerVisionAssignments.motionAndVideo as motion
    destination = "C:/DP/Data/Preprocess/motion/"

    fileProcessing.separateImageAndMask(directory, destination)
    tempImage = cv2.imread("C:\\DP\\Data\\Preprocess\\motion\\1\\1_1.tif")
    for patientDirectory in os.listdir(destination):
        # for name in os.listdir(patientDirectory):
        if ".avi" in patientDirectory:
            continue
        directoryName = destination + patientDirectory
        # tempImage.shape[0], tempImage.shape[1]
        motion.createVideo(patientDirectory, directoryName)

video = "C:/DP/Data/Preprocess/motion/1/1.avi"
# motion.lucasKanadeOpticalFlow(video=video)




import fileProcessing

destination = "C:/DP/Data/Preprocess/motion/"
motion.showImageAndMask(destination)

# showMotionVectors()


# LR = lr.LogisticRegression(input_dim, num_output_classes)




