from PyQt5              import QtWidgets, QtCore
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def StereoDisparity(pathL,pathR):
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            orgimgRcache = cv.imread(pathR)
            # displaying the coordinates
            # on the Shell
            if disparity[y][x]/16 != -1:
                orgimgRcache = cv.imread(pathR)
                cv.circle(orgimgRcache, (x-int(disparity[y][x]/16),y), radius=20, color=(0, 255, 0), thickness=-1)
            cv.imshow("OriginImageR",orgimgRcache)
    
    def show():
        Form = QtWidgets.QWidget()
        mbox = QtWidgets.QMessageBox(Form)
        mbox.critical(Form, 'critical', 'No File selected! Please load image_L and image_R!')
    if (pathL == " ") | (pathR == " ") :
        show()
    else:  
        orgimgL = cv.imread(pathL)
        orgimgR = cv.imread(pathR)
        
        cv.namedWindow('OriginImageL', cv.WINDOW_NORMAL)
        cv.imshow("OriginImageL",orgimgL)
        cv.namedWindow('OriginImageR', cv.WINDOW_NORMAL)
        cv.imshow("OriginImageR",orgimgR)

        imgL = cv.imread(pathL,0)
        imgR = cv.imread(pathR,0)
        stereo = cv.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL,imgR)
        disparitynor = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        plt.imshow(disparitynor,'gray')
        plt.show()
        
        cv.setMouseCallback('OriginImageL', click_event)