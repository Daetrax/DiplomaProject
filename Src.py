# import the necessary packages
import os
import cv2

import numpy as np

desktopDirectory = "D:/DP/Data/train/train/"
notebookDirectory = "C:/DP/Data/train/"

directory = desktopDirectory
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
    import ComputerVisionAssignments.motionAndVideo as motion
    import ComputerVisionAssignments.ImageSort as sort

    for patientDirectory in os.listdir(destination):
        sequence = sort.getPatientSequence(patientDirectory)
        # motion.createVideoFromListWithMask(sequence, "D:/DP/Data/Preprocess/motion/10/10_sequence.avi")
        videopath = destination + patientDirectory + "\\" + patientDirectory + "_sequence.avi"
        motion.createVideoFromList(sequence, videopath)
        print("Video ", patientDirectory, " done.")

video = "D:/DP/Data/Preprocess/motion/1/1_sequence.avi"
# motion.lucasKanadeOpticalFlow(video=video)

import ComputerVisionAssignments.superpixel as sp

sp.showSuperpixelImages(cv2.imread("D:/DP/Data/Preprocess/motion/1/1_1.tif"), 150, mask=cv2.imread("D:/DP/Data/Preprocess/motion/1/1_1_mask.tif"))


import fileProcessing

destination = "D:\\DP\\Data\\Preprocess\\motion\\"
# motion.showImageAndMask(destination)
# fileProcessing.separateImageAndMask("D:\\DP\\Data\\train\\train\\", destination)

# showMotionVectors()





# LR = lr.LogisticRegression(input_dim, num_output_classes)




