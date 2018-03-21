from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import cv2
import ComputerVisionAssignments.filters as flt

def showSuperpixelImages(image, numSegments, mask=None):

    image_gauss = flt.gauss_filter(image)
    image_wiener = flt.wiener_filter_scipy(image)
    image_bilateral = flt.bilateral_filter(image)
    image_median = flt.median_filter(image)
    print("Filtering done")

    segments = slic(image, n_segments=numSegments, sigma=5)

    segments_gauss = slic(image_gauss, n_segments=numSegments, sigma=5)
    print("Gauss segments done")
    segments_wiener = slic(image_wiener, n_segments=numSegments, sigma=5)
    print("Wiener segments done")
    segments_bilateral = slic(image_bilateral, n_segments=numSegments, sigma=5)
    print("Bilateral segments done")
    segments_median = slic(image_median, n_segments=numSegments, sigma=5)
    print("Median segments done")

    print("Segments done")

    if mask is not None:
        print("Mask passed as argument, generating image with contours")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        imageWithContours = image.copy()
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imageWithContours, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Base with label", imageWithContours)
        cv2.moveWindow("Base with label", 600*2, 460)

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))

    plt.axis("off")
    plt.show()
    cv2.imshow("Superpixels base",mark_boundaries(image, segments))

    cv2.imshow("Superpixels gauss", mark_boundaries(image_gauss, segments_gauss))
    cv2.imshow("Superpixels wiener", mark_boundaries(image_wiener, segments_wiener))
    cv2.imshow("Superpixels bilateral", mark_boundaries(image_bilateral, segments_bilateral))
    cv2.imshow("Superpixels median", mark_boundaries(image_median, segments_median))

    cv2.moveWindow("Superpixels base", 0, 0)

    cv2.moveWindow("Superpixels gauss", 600, 0)
    cv2.moveWindow("Superpixels wiener", 600*2, 0)
    cv2.moveWindow("Superpixels bilateral", 0, 460)
    cv2.moveWindow("Superpixels median", 600, 460)

    cv2.waitKey(0)