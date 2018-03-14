import cv2

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def start(directory):
    img = cv2.imread(directory + "1_1.tif")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean, dev = cv2.meanStdDev(img)

    img2 = img.copy()

    img2 = (img2 - mean) / dev

    # img2 = cv2.subtract(img2, mean)
    # img2 = cv2.divide(img2, dev)
    # img3 = cv2.bilateralFilter(img, 10, 300, 300)

    # kernel = np.ones((5,5),np.uint8)
    # img3 = cv2.dilate(img, kernel, iterations=1)
    # img3 = cv2.erode(img3, kernel, iterations=1)

    # guide, radius, esp TODO: how to get guide
    # filter = cv2.ximgproc.createGuidedFilter(img, 13, 70)
    # img3 = filter.filter(img, 13, 70)

    img3 = cv2.medianBlur(img, 3)

    img3 = cv2.fastNlMeansDenoising(img,None,10,7,21)

    cv2.imshow("Before", img)
    cv2.imshow("After", img2)
    cv2.imshow("Guided", img3)
    cv2.waitKey(0)

    # cv2.imshow("Lee filter",lee_filter(img, 20))
    # cv2.waitKey(0)