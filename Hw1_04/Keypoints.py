from PyQt5      import QtWidgets, QtCore
import cv2 as cv
from matplotlib import pyplot as plt

def Keypoints(path1):
    def show():
            Form = QtWidgets.QWidget()
            mbox = QtWidgets.QMessageBox(Form)
            mbox.critical(Form, 'critical', 'No File selected! Please load Image 1!')
    
    if (path1 == " "):
        show()
    else:
        img  = cv.imread(path1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Keypoints
        sift   = cv.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray,None)
        key_image = cv.drawKeypoints(gray,keypoints,img, (0,255,0))
        cv.imwrite('sift_keypoints.jpg',key_image)
        plt.imshow(key_image)
        plt.show()