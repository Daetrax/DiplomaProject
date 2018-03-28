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

def createVideosFromImages(destination):
    import ComputerVisionAssignments.motionAndVideo as motion
    import ComputerVisionAssignments.ImageSort as sort

    for i, patientDirectory in enumerate(os.listdir(destination)):
        sequence = sort.getPatientSequence(patientDirectory)
        # motion.createVideoFromListWithMask(sequence, "D:/DP/Data/Preprocess/motion/10/10_sequence.avi")
        videopath = destination + patientDirectory + "\\" + patientDirectory + "_sequence.avi"
        motion.createVideoFromListWithMask(sequence, videopath)
        print("Video ", patientDirectory, " done.", "  progress: ", i, " / ", len(os.listdir(destination)))

video = "D:/DP/Data/Preprocess/motion/1/1_sequence.avi"
# motion.lucasKanadeOpticalFlow(video=video)

import ComputerVisionAssignments.superpixel as sp

cap = cv2.VideoCapture(video)
ret, frame = cap.read()
centroids = sp.getSuperpixelCentroids(frame, 100)
cap.release()

createVideosFromImages("D:/DP/Data/Preprocess/motion/")

motion.lucasKanadeOpticalFlow(video=video, points_to_track=centroids, filter="gauss")
motion.denseOpticalFlow(video)
sp.showSuperpixelImages(cv2.imread("D:/DP/Data/Preprocess/motion/1/1_1.tif"), 150, mask=cv2.imread("D:/DP/Data/Preprocess/motion/1/1_1_mask.tif"))


import fileProcessing

destination = "D:\\DP\\Data\\Preprocess\\motion\\"
# motion.showImageAndMask(destination)
# fileProcessing.separateImageAndMask("D:\\DP\\Data\\train\\train\\", destination)

# showMotionVectors()





# LR = lr.LogisticRegression(input_dim, num_output_classes)




