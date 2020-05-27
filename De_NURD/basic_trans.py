
#used python packages
import cv2
import math
import numpy as np
import os
import random

class Basic_oper(object):


    def tranfer_frome_cir2rec(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))

        polar_image = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_FILL_OUTLIERS)

        polar_image = polar_image.astype(np.uint8)
        polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return polar_image
    def tranfer_frome_rec2cir(gray):
        H,W = gray.shape
        value = np.sqrt(((H/2.0)**2.0)+((W/2.0)**2.0))
        gray=cv2.rotate(gray,rotateCode = 2) 
        #value = 200
        #circular = cv2.linearPolar(new_frame, (new_frame.shape[1]/2 , new_frame.shape[0]/2), 
        #                               200, cv2.WARP_INVERSE_MAP)
        circular = cv2.linearPolar(gray,(W/2, H/2), value, cv2.WARP_INVERSE_MAP)

        circular = circular.astype(np.uint8)
        #polar_image=cv2.rotate(polar_image,rotateCode = 0) 
        return circular

