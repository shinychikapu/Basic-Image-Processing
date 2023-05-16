import numpy as np
import cv2 as cv

def mean_filter(img_path):
    # image path 
    path = r'cat.jpg'

    # using imread()  
    img = cv.imread(path)
    dst = cv.medianBlur(img,7)

    cv.imshow('image', numpy.hstack((img, dst)))
    cv.waitKey(0);
    cv.destroyAllWindows();
    cv.waitKey(1)