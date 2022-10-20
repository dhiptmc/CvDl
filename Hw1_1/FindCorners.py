from PyQt5              import QtWidgets, QtCore
import numpy as np
import cv2   as cv
import os
import glob

def FindCorners(path):
    def show():
        Form = QtWidgets.QWidget()
        mbox = QtWidgets.QMessageBox(Form)
        mbox.critical(Form, 'critical', 'No File selected! Please load image folder!')
    if path == " ":
        show()
    else:    
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob(os.path.join(path, "*.bmp"))
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (8,11), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (8,11), corners2, ret)
                cv.namedWindow('FindCorners', cv.WINDOW_NORMAL)
                cv.imshow('FindCorners', img)
                cv.waitKey(500)
        cv.destroyAllWindows()