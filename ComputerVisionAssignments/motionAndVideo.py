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

def createVideoFromList(frames, videoname):

    videoPath = videoname

    tempImage = cv2.imread(frames[0])
    height = tempImage.shape[0]
    width = tempImage.shape[1]

    print(width, height)
    video = cv2.VideoWriter(filename=videoPath, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=7, frameSize=(width, height))

    for framename in frames:
        frame = cv2.imread(framename)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def createMaskVideo(frames, videoname):
    """expecting a list of file names and video name"""
    videoPath = videoname

    tempImage = cv2.imread(frames[0])
    height = tempImage.shape[0]
    width = tempImage.shape[1]

    print(width, height)
    video = cv2.VideoWriter(filename=videoPath, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=7,
                            frameSize=(width, height))

    for framename in frames:
        # frame = cv2.imread(framename)

        multichannelMask = cv2.imread(framename.replace(".tif", "_mask.tif"))
        # mask = cv2.imread(framename.replace(".tif", "_mask.tif"), cv2.IMREAD_GRAYSCALE)

        video.write(multichannelMask)
    cv2.destroyAllWindows()
    video.release()

def createVideoFromListWithMask(frames, videoname):

    videoPath = videoname

    tempImage = cv2.imread(frames[0])
    height = tempImage.shape[0]
    width = tempImage.shape[1]

    print(width, height)
    video = cv2.VideoWriter(filename=videoPath, fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=7, frameSize=(width*2, height*2))



    for framename in frames:
        frame = cv2.imread(framename)

        imageWithContours = frame.copy()

        multichannelMask = cv2.imread(framename.replace(".tif", "_mask.tif"))
        mask = cv2.imread(framename.replace(".tif", "_mask.tif"), cv2.IMREAD_GRAYSCALE)
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imageWithContours, contours, -1, (0, 255, 0), 1)

        emptyImg = np.zeros((420, 580, 3), np.uint8)
        combined2 = np.concatenate((imageWithContours, emptyImg), axis=1)

        combined = np.concatenate((frame, multichannelMask), axis=1)
        combined = np.concatenate((combined, combined2), axis=0)
        video.write(combined)
    cv2.destroyAllWindows()
    video.release()


def getFramesDifference(old_frame, new_frame, p0, frame, win_name="Optical flow"):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)

    # Select good points
    good_new = p1
    good_old = p0
    # draw the tracks
    mask = np.zeros_like(frame)

    # frame = new_frame.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.arrowedLine(mask, (c, d), (a, b), [0, 255, 255], thickness=2, tipLength=0.3)
        frame = cv2.circle(frame, (a, b), 3, [0, 255, 255], -1)
    img = cv2.add(frame, mask)
    # cv2.imshow(win_name, img)
    # k = cv2.waitKey(30) & 0xff
    return p1, st, err, img


def switch(filtermode, image):
    import ComputerVisionAssignments.filters as f
    imagecopy = image.copy()
    return {
        'gauss': f.gauss_filter(imagecopy),
        # 'wiener': f.wiener_filter_skimage(imagecopy),
        'bilateral': f.bilateral_filter(imagecopy),
        'median': f.median_filter(imagecopy),
        None: image
    }[filtermode]

def lucasKanadeOpticalFlowCompare(video, filter=None):
    global old_filtered
    import numpy as np
    import cv2, time
    cap = cv2.VideoCapture(video)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=150,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if filter is not None:
        old_filtered = switch(filter, old_gray)
        p0_filtered = cv2.goodFeaturesToTrack(old_filtered, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    counter = 0
    while counter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Now update the previous frame and previous points
        p1, st, err, diff_img = getFramesDifference(old_gray, frame_gray, p0, frame, win_name="Base")

        if filter is not None:
            framecopy = frame.copy()
            frame_filtered = switch(filter, frame_gray)
            p1_filtered, st_filtered, err_filtered, diff_filtered = \
                getFramesDifference(old_filtered, frame_filtered, p0_filtered, framecopy, win_name="Filtered")

        good_new = p1
        old_gray = frame_gray.copy()

        if filter is not None:
            good_new_filtered = p1_filtered
            old_filtered = frame_filtered.copy()
            p0_filtered = good_new_filtered.reshape(-1, 1, 2)
            name = filter + " " + filter
            cv2.imshow(name, diff_filtered)
            cv2.moveWindow(name, diff_filtered.shape[1], 0)

        cv2.imshow("base", diff_img)
        p0 = good_new.reshape(-1, 1, 2)
        cv2.waitKey(1)
        time.sleep(0.325)
    cv2.destroyAllWindows()
    cap.release()


def lucasKanadeOpticalFlow(video, filter=None, points_to_track=None, delay=0.325):
    global old_filtered
    import numpy as np
    import cv2, time
    cap = cv2.VideoCapture(video)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=150,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = switch(filter, old_gray)

    if points_to_track is not None:
        p0 = np.asarray(points_to_track, dtype=np.float32)
    else:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    counter = 0
    while counter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = switch(filter, frame_gray)

        # Now update the previous frame and previous points
        p1, st, err, diff_img = getFramesDifference(old_gray, frame_gray, p0, frame, win_name="Base")

        good_new = p1
        old_gray = frame_gray.copy()

        cv2.imshow("base", diff_img)
        p0 = good_new.reshape(-1, 1, 2)
        cv2.waitKey(0)
        time.sleep(delay)
    cv2.destroyAllWindows()
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


def draw_flow(img, flow, threshold, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    ffx, ffy = [], []
    for a, b in zip(fx, fy):
        if (abs(a) > threshold or abs(b) > threshold):
            ffx.append(a)
            ffy.append(b)
        else:
            ffx.append(0)
            ffy.append(0)
    ffx = np.asarray(ffx, dtype=np.float32)
    ffy = np.asarray(ffy, dtype=np.float32)
    # print(ffx, "\n\n", ffy)
    lines = np.vstack([x, y, x+ffx, y+ffy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    lines2 = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines2 = np.int32(lines2 + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img.copy()
    vis2 = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 255))
    cv2.polylines(vis2, lines2, 0, (0, 255, 255))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 255), -1)
        cv2.circle(vis2, (x1, y1), 1, (0, 255, 255), -1)
    return vis, vis2


def drawFlow(img, flow, step=16):
    height, width = img.shape[:2]
    y, x = np.mgrid[step / 2:height:step, step / 2:width:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    result = img.copy()

    for i in range(0, height, step):
        for j in range(0, width, step):
            k, l = flow[i, j]
            k = (i - k).astype('float32')
            l = (j - l).astype('float32')
            x, y = float(i), float(j)
            cv2.arrowedLine(result, (x, y), (k, l), (0, 255, 255), 2)

    # cv2.arrowedLine(result, (x, y), (x + fx, y + fy), (0, 255, 255), 2)
    return result




def denseOpticalFlow(video, threshold=0):
    import cv2 as cv
    import numpy as np
    import time
    cap = cv.VideoCapture(video)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    counter = 1

    video_writer = cv2.VideoWriter(filename="flow.avi", fourcc=cv2.VideoWriter_fourcc(*"MJPG"), fps=7, frameSize=(580, 420))

    while counter < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        counter += 1

        flow_thr, flow_image = draw_flow(img=frame2, flow=flow, threshold=threshold)
        cv2.imshow("Flow", flow_image)
        cv2.imshow("Flow thresholded", flow_thr)

        cv2.moveWindow("Flow", 50, 50)
        cv2.moveWindow("Flow thresholded", 650, 50)

        # cv2.waitKey(0)

        video_writer.write(flow_image)

        cv.waitKey(1)
        prvs = next
        time.sleep(0.225)
    cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()
    video_writer.release()


def showImageAndMask(directory):
    import time

    for i, dirName in enumerate(os.listdir(directory)):

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
            # cv2.imshow("Combined", vis)
            #
            # cv2.waitKey(1)
            # time.sleep(0.25)
        createVideo(frames, patientDirectory + "/" + dirName + "_withContours.avi")
        print("created video: ", dirName, "  progress: ", i, " / ", len(os.listdir(directory)))
        # cv2.waitKey(0)
