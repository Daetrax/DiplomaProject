# import the necessary packages
import os
import cv2

import numpy as np

desktopDirectory = "D:/DP/Data/train/train/"
notebookDirectory = "C:/DP/Data/train/"

directory = desktopDirectory
dest = "D:/DP/Data/Preprocess_1/train/"
# len = len(os.listdir(directory))

def logisticRegression():
    """temporary function, refactor later"""

    import NN.LogisticRegression as lr
    # Define the data dimensions
    input_dim = 841
    num_output_classes = 2

    # deseralizer = MyDeserializer.MyDeserializer(directory=dataDirectory, streams=[dict(name='Features', shape=(361,)), dict(name='Labels', shape=(2,))])
    #
    # deseralizer.createDictionary(dataDirectory)
    #
    # deseralizer.get_chunk(2)

    LR = lr.LogisticRegression(input_dim, num_output_classes)



import ComputerVisionAssignments.motionAndVideo as motion

def createVideosFromImages(destination):
    """temporary function, refactor later"""

    import ComputerVisionAssignments.motionAndVideo as motion
    import ComputerVisionAssignments.ImageSort as sort

    for i, patientDirectory in enumerate(os.listdir(destination)):
        sequence = sort.getPatientSequence(patientDirectory)
        # motion.createVideoFromListWithMask(sequence, "D:/DP/Data/Preprocess/motion/10/10_sequence.avi")
        videopath = destination + patientDirectory + "\\" + patientDirectory + "_sequence.avi"
        motion.createVideoFromListWithMask(sequence, videopath)

        videopath_mask = destination + patientDirectory + "\\" + patientDirectory + "_sequence_mask.avi"
        motion.createMaskVideo(sequence, videopath_mask)

        print("Video ", patientDirectory, " done.", "  progress: ", i, " / ", len(os.listdir(destination)))

def getSortedFrames(patientId, directory="D:/DP/Data/Preprocess/motion/"):
    """temporary function, refactor later"""

    import ComputerVisionAssignments.ImageSort as sort
    patientDirectory = directory + str(patientId) + "/"
    return sort.getPatientSequence(patientDirectory)


video = "D:/DP/Data/Preprocess/motion/1/1_sequence.avi"
video_mask = "D:/DP/Data/Preprocess/motion/1/1_sequence_mask.avi"

# motion.lucasKanadeOpticalFlow(video=video)


def dbscan():
    """temporary function, refactor later"""
    import ComputerVisionAssignments.clustering as cl
    import ComputerVisionAssignments.DetectorsAndDescriptors as dtct

    sequence = getSortedFrames(1)

    for framepath in sequence:
        frame = cv2.imread(framepath)
        frame_mask = cv2.imread(framepath.replace(".tif", "_mask.tif"))

        filtered_img = sp.switch(image=frame, filtermode="gauss")
        sift_points = dtct.siftPoints(filtered_img)
        keyps = cv2.drawKeypoints(filtered_img, sift_points, filtered_img, color=(0, 255, 255))
        cv2.imshow("Keyps", filtered_img)
        cv2.waitKey(0)
        cl.dbscan(sift_points, frame, frame_mask)

def superpixels():
    """temporary function, refactor later"""
    import ComputerVisionAssignments.superpixel as sp

    sequence = getSortedFrames(1)

    for framepath in sequence:
        frame = cv2.imread(framepath)
        frame_mask = cv2.imread(framepath.replace(".tif", "_mask.tif"))

        # track centroids of superpixel with lucas kanade algorithm
        centroids = sp.getSuperpixelCentroids(image=frame, numSegments=100, debug=False, filter="gauss")
        motion.lucasKanadeOpticalFlow(video=video, points_to_track=centroids, filter="gauss")

        # show superpixels with various filters
        sp.showSuperpixelImages(image=frame, numSegments=150, mask=frame_mask)

