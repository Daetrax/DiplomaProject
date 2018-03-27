import numpy as np
import cv2
import random


selRoi = 0
top_left = [160, 213]
bottom_right = [320, 426]
first_time = 1

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


def findDistance(r1, c1, r2, c2):
    d = (r1 - r2) ** 2 + (c1 - c2) ** 2
    d = d ** 0.5
    return d


# main function
cv2.namedWindow('tracker')
import time

cap = cv2.VideoCapture("D:\\DP\\Data\\Preprocess\\motion\\1\\1_sequence.avi")

def is_square(n):
    import math
    return math.sqrt(n).is_integer()

def getOldCorners():
    counter = 1
    while counter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        _, frame = cap.read()
        # -----Drawing Stuff on the Image
        # cv2.putText(frame, 'Press a to start tracking', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(60, 100, 75),
        #             thickness=3)
        # cv2.rectangle(frame, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]),
        #               color=(100, 255, 100), thickness=4)

        print("Frame ", counter)
        counter += 1
        # -----Finding ROI and extracting Corners
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # roi = frameGray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]  # selecting roi
        roi = frameGray.copy()
        new_corners = cv2.goodFeaturesToTrack(roi, 150, 0.01, 10)  # find corners

        # -----converting to complete image coordinates (new_corners)

        # new_corners[:, 0, 0] = new_corners[:, 0, 0] + top_left[1]
        # new_corners[:, 0, 1] = new_corners[:, 0, 1] + top_left[0]

        # -----drawing the corners in the original image
        for corner in new_corners:
            cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, (0, 255, 0))

        # -----old_corners and oldFrame is updated
        oldFrameGray = frameGray.copy()
        old_corners = new_corners.copy()

        cv2.imshow('tracker', frame)

        a = cv2.waitKey(5)
        time.sleep(0.125)
        if counter == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            cv2.destroyAllWindows()
            cap.release()
            return oldFrameGray, old_corners

def is_in_segment(feature, segment):
    if (segment["startX"] < feature[0][1] < segment["startX"] + segment["X"] and
            segment["startY"] < feature[0][0] < segment["startY"] + segment["Y"]):
        return True
    return False


def trackSegmentMotion(old_image, new_image, parts = 16):
    """Images have to be grayscale"""
    import math
    old_features = cv2.goodFeaturesToTrack(old_image, mask=None, **feature_params)
    new_features, st, err = cv2.calcOpticalFlowPyrLK(old_image, new_image, old_features, None, **lk_params)

    if not is_square(parts):
        print("Parts are not a square number")
        return

    block_width = new_image.shape[1] / 4
    block_height = new_image.shape[0] / 4
    startX = 1
    startY = 1

    for i in range(1, parts + 1, 1):
        startX = i % math.sqrt(parts)

        if startX == 0:
            startX = 1
            startY += 1

        segment = {
            "startX": startX,  # height
            "startY": startY,  # width
            "X": block_width,
            "Y": block_height
        }

        if not is_in_segment(new_features, segment):
            continue
        # TODO: continue here?
        r_add, c_add = 0, 0
        for corner in new_features:

            r_add = r_add + corner[0][1]
            c_add = c_add + corner[0][0]
        centroid_row = int(1.0 * r_add / len(new_features))
        centroid_col = int(1.0 * c_add / len(new_features))
        # draw centroid
        cv2.circle(new_image, (int(centroid_col), int(centroid_row)), 5, (255, 0, 0))
        # add only those corners to new_corners_updated which are at a distance of 30 or less
        new_corners_updated = new_features.copy()
        tobedel = []
        for index in range(len(new_features)):
            if findDistance(new_features[index][0][1], new_features[index][0][0], int(centroid_row),
                            int(centroid_col)) > 90:
                tobedel.append(index)
        new_corners_updated = np.delete(new_corners_updated, tobedel, 0)

        # drawing the new points
        for corner in new_corners_updated:
            cv2.circle(new_image, (int(corner[0][0]), int(corner[0][1])), 5, (0, 255, 0))
        if len(new_corners_updated) < 10:
            print('OBJECT LOST, Reinitialize for tracking')
            break
        # finding the min enclosing circle
        ctr, rad = cv2.minEnclosingCircle(new_corners_updated)

        cv2.circle(new_image, (int(ctr[0]), int(ctr[1])), int(rad), (0, 0, 255), thickness=5)


def trackMotion():
    cap = cv2.VideoCapture("D:\\DP\\Data\\Preprocess\\motion\\1\\1_sequence.avi")

    # oldFrameGray, old_corners = getOldCorners()

    counter = 1
    while counter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        'Now we have oldFrame,we can get new_frame,we have old corners and we can get new corners and update accordingly'
        print("Second cycle frame ", counter)

        # read new frame and cvt to gray
        ret, frame = cap.read()

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if counter == 1:
            oldFrameGray = frameGray.copy()
            old_corners = cv2.goodFeaturesToTrack(frameGray, 100, 0.01, 10)

        counter += 1
        # finding the new tracked points
        new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, frameGray, old_corners, None, **lk_params)

        # ---pruning far away points:
        # first finding centroid
        r_add, c_add = 0, 0
        for corner in new_corners:
            r_add = r_add + corner[0][1]
            c_add = c_add + corner[0][0]
        centroid_row = int(1.0 * r_add / len(new_corners))
        centroid_col = int(1.0 * c_add / len(new_corners))
        # draw centroid
        cv2.circle(frame, (int(centroid_col), int(centroid_row)), 5, (255, 0, 0))
        # add only those corners to new_corners_updated which are at a distance of 30 or less
        new_corners_updated = new_corners.copy()
        tobedel = []
        for index in range(len(new_corners)):
            if findDistance(new_corners[index][0][1], new_corners[index][0][0], int(centroid_row),
                            int(centroid_col)) > 90:
                tobedel.append(index)
        new_corners_updated = np.delete(new_corners_updated, tobedel, 0)

        # drawing the new points
        for corner in new_corners_updated:
            cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, (0, 255, 0))
        if len(new_corners_updated) < 10:
            print('OBJECT LOST, Reinitialize for tracking')
            break
        # finding the min enclosing circle
        ctr, rad = cv2.minEnclosingCircle(new_corners_updated)

        cv2.circle(frame, (int(ctr[0]), int(ctr[1])), int(rad), (0, 0, 255), thickness=5)

        # updating old_corners and oldFrameGray
        oldFrameGray = frameGray.copy()
        old_corners = new_corners_updated.copy()

        # showing stuff on video
        cv2.putText(frame, 'Tracking Integrity : Excellent %04.3f' % random.random(), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color=(200, 50, 75), thickness=3)
        cv2.imshow('tracker', frame)

        a = cv2.waitKey(5)
        time.sleep(0.125)
        if a == 27:
            cv2.destroyAllWindows()
            cap.release()
        elif a == 97:
            break
    cv2.destroyAllWindows()
    cap.release()


trackMotion()
print("Finished, waiting for key input.")
cv2.waitKey(0)
cv2.destroyAllWindows()
