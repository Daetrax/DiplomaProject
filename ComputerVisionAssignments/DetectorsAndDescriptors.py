import cv2
import numpy as np

def siftPoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    sift_points = sift.detect(gray, None)
    return sift_points
    # temp = []
    # temp = [ (kp.pt[0], kp.pt[1]) for kp in sift_points ]
    # temp = np.asarray(temp, dtype=np.float32)
