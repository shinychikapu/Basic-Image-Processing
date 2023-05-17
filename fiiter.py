import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

def median_filter(img_path):
    '''This function take in a path to the image on the computer and apply median filtering to it

    Attribute
    -------
    img_path (String): the path to the image on the computer
    '''

    print("Median Filtering")

    img = cv.imread(img_path)
    dst = cv.medianBlur(img,3)

    cv.imshow('image', np.hstack((img, dst)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

def mean_filter(img_path):
    '''This function take in a path to the image on the computer and apply mean filtering to it

    Attribute
    -------
    img_path (String): the path to the image on the computer
    '''

    print("Mean Filtering")

    img = cv.imread(img_path)

    new_img = cv.blur(img,(5,5)) 

    cv.imshow('image', np.hstack((img, new_img)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

def sharpen(img_path):
    '''This function take in a path to the image on the computer and apply sharpening to the photo

    Attribute
    -------
    img_path (String): the path to the image on the computer
    '''

    print("Sharpening")

    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Storing the img dimension in a tuplet
    #h, w, c = img.shape
    #Reshape the img
    #img = img.reshape(c, h, w)
    #convert the image to tensor
    img_ = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    #img_.unsqueeze_(0)
    

    #Define the sharpening filter, this one is usually used for sharpening img
    #Laplacian filter
    sharpen_filter = torch.tensor([[0, 1, 0], 
                                [1, -4.2, 1],
                                [0, 1, 0]]).unsqueeze(0).unsqueeze(0)

    #Reshape the dimension
    #sharpen_filter = sharpen_filter.reshape(1, 1, 3, 3)
    #Convert the ndarray to torch.tensor

    #Defining the convolution
    sharpen = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size= 3, bias= False)
    #Set the filter by changing the weight
    sharpen.weight = nn.Parameter(sharpen_filter.float(), requires_grad= False)

    #Apply the convolution to the image
    sharpenedImg = sharpen(img_).squeeze()
    sharpenedImg = sharpenedImg.detach().cpu().numpy()
    #sharpenedImg = F.conv2d(img_, sharpen_filter)

    cv.imshow('image', sharpenedImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
