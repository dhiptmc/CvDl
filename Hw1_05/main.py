from PyQt5       import QtWidgets, QtCore, QtGui
import sys
from predict import predict
from trainimages import trainimages
from modelstructure import model
from dataAugmentation import augmentation
from predict import predict
import numpy as np
import cv2 as cv

fip = " "

#####Functions#####
def openFile():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()  # 選取單一檔案
    global fip
    fip = str(filePath)
    print(filePath, filterType )
    img = QtGui.QPixmap(fip)                # 加入圖片
    img = img.scaled(320,320)               # 調整圖片大小為 320x320
    scene.addPixmap(img)                    # 將圖片加入 scene

def showResult():
    img_result1 = cv.imread("output/plot3.png")
    cv.imshow('Result1', img_result1)
    img_result2 = cv.imread("output/screenshot3.png")
    cv.imshow('Result2', img_result2)   
    cv.waitKey(0)
    cv.destroyAllWindows()


style ='''
    QPushButton {
        font-size:16px;
        color: #808000;
        background: #FAEBD7;
        border: 2px solid #fff;
    }
    QPushButton:hover {
        color: #FAEBD7;
        background: #808000;
    }
'''

#####main#####
app = QtWidgets.QApplication(sys.argv)
Form = QtWidgets.QWidget()
Form.setWindowTitle('2022CvDl Hw1')         # 設定標題
Form.resize(1120, 700)                      # 設定長寬尺寸
Form.setStyleSheet('background:#BDB76B;')   # 使用網頁 CSS 樣式設定背景

#####Column 0#####

label0 = QtWidgets.QLabel(Form)
label0.setText('VGG19 test')
label0.setGeometry(0, 30, 250, 50)
label0.setAlignment(QtCore.Qt.AlignCenter)  # 對齊方式
label0.setWordWrap(True)
label0.setStyleSheet('''
    color:#FAEBD7;
    font-size:25px;
    font-weight:bold;
''')

btn01 = QtWidgets.QPushButton(Form)
btn01.setText('Load Image')   
btn01.setGeometry(100,100,200,50)
btn01.setStyleSheet((style))
btn01.clicked.connect(openFile)

btn02 = QtWidgets.QPushButton(Form)    
btn02.setText('1. Show Train Images')      
btn02.setGeometry(100,200,200,50)
btn02.setStyleSheet((style))
btn02.clicked.connect(trainimages)

btn03 = QtWidgets.QPushButton(Form)    
btn03.setText('2. Show Model Structure')      
btn03.setGeometry(100,300,200,50)
btn03.setStyleSheet((style))
btn03.clicked.connect(model)

btn04 = QtWidgets.QPushButton(Form)    
btn04.setText('3. Show Data Augmentation')      
btn04.setGeometry(100,400,200,50)
btn04.setStyleSheet((style))
btn04.clicked.connect(augmentation)

btn05 = QtWidgets.QPushButton(Form)    
btn05.setText('4. Show Accuracy and Loss')      
btn05.setGeometry(100,500,200,50)
btn05.setStyleSheet((style))
btn05.clicked.connect(showResult)

btn06 = QtWidgets.QPushButton(Form)    
btn06.setText('5. Inference')      
btn06.setGeometry(100,600,200,50)
btn06.setStyleSheet((style))
btn06.clicked.connect(lambda:predict(fip))

grview = QtWidgets.QGraphicsView(Form)  # 加入 QGraphicsView
grview.setGeometry(500, 200, 350, 350)  # 設定 QGraphicsView 位置與大小
scene = QtWidgets.QGraphicsScene()      # 加入 QGraphicsScene
scene.setSceneRect(0, 0, 320, 320)      # 設定 QGraphicsScene 位置與大小
grview.setScene(scene)                  # 設定 QGraphicsView 的場景為 scene

Form.show()
sys.exit(app.exec_())
