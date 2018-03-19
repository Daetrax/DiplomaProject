import cv2, os, numpy as np


def createVideosFromDirectory(videoName, directory, imageWidth, imageHeight):

    videoPath = directory + "/" + videoName + ".avi"

    video = cv2.VideoWriter(filename=videoPath, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=7, frameSize=(580, 420))
    counter = 1

    for name in os.listdir(directory):
        if ".avi" in name:
            continue

        filename = directory + "/" + name.split("_")[0] + "_" + str(counter) + ".tif"
        counter += 1

        image = cv2.imread(filename)
        print(filename)
        video.write(image.astype('uint8'))



    cv2.destroyAllWindows()
    video.release()

def createVideo(frames, videoname):

    videoPath = videoname

    height = frames[0].shape[0]
    width = frames[0].shape[1]
    print(width, height)
    video = cv2.VideoWriter(filename=videoPath, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=7, frameSize=(width, height))

    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def lucasKanadeOpticalFlow(video):
    import numpy as np
    import cv2 as cv
    import time
    # cap = cv.VideoCapture('slow.flv')
    cap = cv.VideoCapture(video)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=200,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    drawingMask = np.zeros_like(old_frame)

    # retval, mask = cv.threshold(old_gray, -1, 255, cv.THRESH_OTSU)
    # mask = cv.medianBlur(mask, 11)
    # old_gray = cv.bitwise_and(old_gray, old_gray, mask=mask)
    # cv.imshow("Mask", mask)
    # cv.imshow("MaskBlurred", mask2)
    # cv.waitKey(0)
    counter = 1
    print(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
    while counter < int(cap.get(cv.CAP_PROP_FRAME_COUNT)):
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        retval, mask = cv.threshold(frame_gray, -1, 255, cv.THRESH_OTSU)
        mask = cv.medianBlur(mask, 11)

        # frame_gray = cv.bitwise_and(frame_gray, frame_gray, mask=mask)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        print(counter)
        counter += 1

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            drawingMask = cv.line(drawingMask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
        print("Shapes: ", drawingMask.shape, frame.shape, "\nSizes: ", drawingMask.size, frame.size)

        img = cv.add(frame, drawingMask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        # If no movement was detected, copy previous points .
        if good_new.size == 0 or good_old.size == 0:
            continue
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        time.sleep(0.125)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cap.release()

def showMotionVectors(directory):
    import fileProcessing
    # destination = "D:/DP/Data/Preprocess/motion/"
    destination = directory
    for patientDirectory in os.listdir(destination):
        # for name in os.listdir(patientDirectory):
        videoPath = destination + patientDirectory + "/" + patientDirectory + ".avi"
        lucasKanadeOpticalFlow(video=videoPath)

def detectBasicMotion(video, min_area):
    import imutils, datetime

    firstFrame = None
    camera = cv2.VideoCapture(video)

    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        (grabbed, frame) = camera.read()
        text = "Unoccupied"

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        #                              cv2.CHAIN_APPROX_SIMPLE)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"
            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)

            cv2.moveWindow("Security Feed", 20, 20)
            cv2.moveWindow("Thresh", 600, 20)
            cv2.moveWindow("Frame Delta", 20, 440)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

def denseOpticalFlow(video):
    import cv2 as cv
    import numpy as np
    import time
    cap = cv.VideoCapture(video)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while (1):
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame2)
            cv.imwrite('opticalhsv.png', bgr)
        prvs = next
        time.sleep(0.125)
    cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()


def showImageAndMask(directory):
    import time

    for dirName in os.listdir(directory):

        patientDirectory = directory + dirName
        frames = []
        for name in os.listdir(patientDirectory):
            if "mask" in name or ".avi" in name:
                continue
            image = cv2.imread(patientDirectory  + "/" + name)
            multichannelMask = cv2.imread(patientDirectory + "/" + name.replace(".tif", "") + "_mask.tif")
            mask = cv2.imread(patientDirectory + "/" + name.replace(".tif", "") + "_mask.tif", cv2.IMREAD_GRAYSCALE)

            # cv2.imshow("Image", image)
            # cv2.imshow("Mask", mask)
            imageWithContours = image.copy()

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imageWithContours, contours, -1, (0, 255, 0), 1)

            # cv2.imshow("With contours", imageWithContours)

            # cv2.moveWindow("Image", 20, 20)
            # cv2.moveWindow("Mask", 600, 20)
            # cv2.moveWindow("With contours", 20, 440)
            emptyImg = np.zeros((420, 580, 3), np.uint8)
            vis2 = np.concatenate((imageWithContours, emptyImg), axis=1)

            vis = np.concatenate((image, multichannelMask), axis=1)
            vis = np.concatenate((vis, vis2), axis=0)
            frames.append(vis)
            cv2.imshow("Combined", vis)

            cv2.waitKey(1)
            time.sleep(0.25)
        createVideo(frames, patientDirectory + "/" + dirName + "_withContours.avi")
        cv2.waitKey(0)
