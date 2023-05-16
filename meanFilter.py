import numpy as np
import cv2 as cv

def mean_filter(img_path):

    # using imread()  
    img = cv.imread(img_path)
    dst = cv.medianBlur(img,7)
    cv.imshow('image', np.hstack((img, dst)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

