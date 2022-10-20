from PyQt5      import QtWidgets, QtCore
import cv2 as cv
from matplotlib import pyplot as plt

def matchedKeypoints(path1,path2):
    def show():
            Form = QtWidgets.QWidget()
            mbox = QtWidgets.QMessageBox(Form)
            mbox.critical(Form, 'critical', 'No File selected! Please load Image 1 and Image 2!')

    if (path1 == " ") | (path2 == " ") :
        show()
    else:
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        
        img1 = cv.imread(path1)
        img2 = cv.imread(path2)
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(gray1,None)
        kp2, des2 = sift.detectAndCompute(gray2,None)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatches(gray1,kp1,gray2,kp2,matches[ :60],None,(255,255,0),(0,255,0),flags=0)
        plt.imshow(img3)
        plt.show()