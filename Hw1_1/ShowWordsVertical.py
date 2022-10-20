from PyQt5              import QtWidgets, QtCore
from ast import Break, If
import numpy as np
import cv2   as cv
import os
import glob

def ShowWordsVertical(path,word):
    def draw(img, imgpts, ch):
        imgpts = np.int32(imgpts).reshape(-1,2)
        # draw ground floor in green
        for i in range (len(ch)):
            for j in range (2):
                if   (ch[i][j]==[2, 2, 0]).all():
                    ch[i][j][0:2] = imgpts[0]
                elif (ch[i][j]==[1, 2, 0]).all():
                    ch[i][j][0:2] = imgpts[1]
                elif (ch[i][j]==[0, 2, 0]).all():
                    ch[i][j][0:2] = imgpts[2]
                elif (ch[i][j]==[2, 1, 0]).all():
                    ch[i][j][0:2] = imgpts[3]
                elif (ch[i][j]==[1, 1, 0]).all():
                    ch[i][j][0:2] = imgpts[4]
                elif (ch[i][j]==[0, 1, 0]).all():
                    ch[i][j][0:2] = imgpts[5]
                elif (ch[i][j]==[2, 0, 0]).all():
                    ch[i][j][0:2] = imgpts[6]
                elif (ch[i][j]==[1, 0, 0]).all():
                    ch[i][j][0:2] = imgpts[7]
                else:
                    ch[i][j][0:2] = imgpts[8]

        for i in range (len(ch)):
            img = cv.line(img, tuple(ch[i][0][0:2]), tuple(ch[i][1][0:2]),(0,0,255),5)
        
        return img

    def show():
        Form = QtWidgets.QWidget()
        mbox = QtWidgets.QMessageBox(Form)
        mbox.critical(Form, 'critical', 'No File selected! Please image folder!')
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
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        filepath = path + "/Q2_lib/alphabet_lib_onboard.txt"
        fs = cv.FileStorage(filepath, cv.FILE_STORAGE_READ)
        
        #axis = np.float32([[ 0, 7, 0], [1, 7, 0], [2, 7, 0], [2, 8, 0], [2, 9, 0], [1, 9, 0], [ 0, 9, 0], [0, 8, 0]])
        points0 = np.float32([[ 2, 9, -2], [2, 8, -2], [2, 7, -2], [2, 9, -1], [2, 8, -1], [2, 7, -1], [ 2, 9, 0], [2, 8, 0], [2, 7, 0]])
        points1 = np.float32([[ 2, 6, -2], [2, 5, -2], [2, 4, -2], [2, 6, -1], [2, 5, -1], [2, 4, -1], [ 2, 6, 0], [2, 5, 0], [2, 4, 0]])
        points2 = np.float32([[ 2, 3, -2], [2, 2, -2], [2, 1, -2], [2, 3, -1], [2, 2, -1], [2, 1, -1], [ 2, 3, 0], [2, 2, 0], [2, 1, 0]])
        points3 = np.float32([[ 5, 9, -2], [5, 8, -2], [5, 7, -2], [5, 9, -1], [5, 8, -1], [5, 7, -1], [ 5, 9, 0], [5, 8, 0], [5, 7, 0]])
        points4 = np.float32([[ 5, 6, -2], [5, 5, -2], [5, 4, -2], [5, 6, -1], [5, 5, -1], [5, 4, -1], [ 5, 6, 0], [5, 5, 0], [5, 4, 0]])
        points5 = np.float32([[ 5, 3, -2], [5, 2, -2], [5, 1, -2], [5, 3, -1], [5, 2, -1], [5, 1, -1], [ 5, 3, 0], [5, 2, 0], [5, 1, 0]])


        for fname in glob.glob(os.path.join(path, "*.bmp")):
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (8,11),None)
            
            name = fname.replace(path,"")
            name = name.replace("\\","")
            name = name.replace(".bmp","")

            if ret == True:


                for i in range (6):
                    # project 3D points to image plane
                    if i == 0:
                        points = points0
                    elif i == 1:
                        points = points1
                    elif i == 2:
                        points = points2 
                    elif i == 3:
                        points = points3 
                    elif i == 4:
                        points = points4 
                    else:
                        points = points5               

                    imgpts, jac = cv.projectPoints(points, rvecs[int(name)-1], tvecs[int(name)-1], mtx, dist)

                    if (len(word)) == i:
                        break
                    else:
                        ch = fs.getNode("%s" %word[i]).mat()
                        draw(img,imgpts,ch)

                cv.namedWindow('ShowWordsVertically', cv.WINDOW_NORMAL)
                cv.imshow('ShowWordsVertically',img)
                cv.waitKey(1000)

        cv.destroyAllWindows()