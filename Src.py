# import the necessary packages
import os
import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

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
    import ComputerVisionAssignments.superpixel as sp

    import ComputerVisionAssignments.clustering as cl
    import ComputerVisionAssignments.DetectorsAndDescriptors as dtct

    sequence = getSortedFrames(1)
    success_rate = []

    for i, framepath in enumerate(sequence):
        frame = cv2.imread(framepath)
        frame_mask = cv2.imread(framepath.replace(".tif", "_mask.tif"))

        filtered_img = sp.switch(image=frame, filtermode="gauss")
        sift_points = dtct.siftPoints(filtered_img)
        # keyps = cv2.drawKeypoints(filtered_img, sift_points, filtered_img, color=(0, 255, 255))
        # cv2.imshow("Keyps", filtered_img)
        # cv2.waitKey(0)
        group_hits, maximum_hits = cl.dbscan(sift_points, frame, frame_mask)

        print(i, " / ", len(sequence))

        a = np.array(group_hits, dtype=np.float)
        b = np.array(maximum_hits, dtype=np.float)

        success_rate.append(a / b)
    return success_rate


def superpixels():
    """temporary function, refactor later"""
    import ComputerVisionAssignments.superpixel as sp

    sequence = getSortedFrames(1)

    for framepath in sequence:
        frame = cv2.imread(framepath)
        frame_mask = cv2.imread(framepath.replace(".tif", "_mask.tif"))

        # track centroids of superpixel with lucas kanade algorithm
        # centroids = sp.getSuperpixelCentroids(image=frame, numSegments=100, debug=False, filter="gauss")
        # motion.lucasKanadeOpticalFlow(video=video, points_to_track=centroids, filter="gauss")

        # show superpixels with various filters
        sp.showSuperpixelImages(image=frame, numSegments=250, mask=frame_mask, framepath=framepath)
import ComputerVisionAssignments.filters as flt

motion.denseOpticalFlow(video)
motion.lucasKanadeOpticalFlow(video, filter="gauss")
sequence = getSortedFrames(1)

for framepath in sequence:
    frame = cv2.imread(framepath)
    frame_mask = cv2.imread(framepath.replace(".tif", "_mask.tif"))

    filtered = flt.wiener_filter_skimage(frame)
    cv2.imshow("Original", frame)
    cv2.imshow("Filtered", filtered)
    cv2.waitKey(0)


    mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Image", frame)
    cv2.waitKey(0)







success_rate = dbscan()
global_average = 0

counter = 0
for array in success_rate:
    if len(array) == 1 or np.count_nonzero(array) == 0:
        counter += 1
        continue

    non_cluster = array[-1]
    array = array[:-1]
    local_average = 0
    local_counter = 0
    for i, ratio in enumerate(array):
        if ratio > 0.01:
            local_average += ratio
        else:
           local_counter += 1
    print()
    # df = pd.DataFrame(array)
    #
    # # you get ax from here
    # ax = df.plot()
    # type(ax)  # matplotlib.axes._subplots.AxesSubplot
    #
    # # manipulate
    # vals = ax.get_yticks()
    # ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
    # plt.show()

    # plt.savefig('D:\\DP\\Data\\MyFigure.jpg')
    print(array)
    l_len = (len(array) - local_counter)
    if l_len <= 0:
        l_len = 1
    local_average /= l_len
    global_average += local_average



global_average /= (len(success_rate) - counter)


print("Global average is: ", global_average)
print("All zeros(no clusters or no hits): ", counter)