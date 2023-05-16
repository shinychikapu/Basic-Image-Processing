import numpy as np
import cv2 as cv

def median_filter(img_path):
    print("Median Filtering")

    img = cv.imread(img_path)
    dst = cv.medianBlur(img,3)

    cv.imshow('image', np.hstack((img, dst)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

def mean_filter(img_path):
    print("Mean Filtering")
    
    img = cv.imread(img_path)

    new_img = cv.blur(img,(5,5)) 

    cv.imshow('image', np.hstack((img, new_img)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
