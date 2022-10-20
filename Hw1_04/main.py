from PyQt5            import QtWidgets, QtCore
import sys
from Keypoints        import Keypoints
from matchedKeypoints import matchedKeypoints
fip1 = " "
fip2 = " "

#####Functions#####
def openFile1():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()  # 選取單一檔案
    global fip1
    fip1 = str(filePath)
    print(filePath, filterType )

def openFile2():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()  # 選取單一檔案
    global fip2
    fip2 = str(filePath)    
    print(filePath, filterType )

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
Form.resize(400, 640)                       # 設定長寬尺寸
Form.setStyleSheet('background:#BDB76B;')   # 使用網頁 CSS 樣式設定背景

#####Column 0#####

label0 = QtWidgets.QLabel(Form)
label0.setText('SIFT')
label0.setGeometry(0, 70, 250, 50)
label0.setAlignment(QtCore.Qt.AlignCenter)  # 對齊方式
label0.setWordWrap(True)
label0.setStyleSheet('''
    color:#FAEBD7;
    font-size:25px;
    font-weight:bold;
''')

btn01 = QtWidgets.QPushButton(Form)
btn01.setText('Load Image 1')   
btn01.setGeometry(100,150,200,50)
btn01.setStyleSheet((style))
btn01.clicked.connect(openFile1)

btn02 = QtWidgets.QPushButton(Form)    
btn02.setText('Load Image 2')      
btn02.setGeometry(100,250,200,50)
btn02.setStyleSheet((style))
btn02.clicked.connect(openFile2)

btn03 = QtWidgets.QPushButton(Form)    
btn03.setText('4.1 Keypoints')      
btn03.setGeometry(100,350,200,50)
btn03.setStyleSheet((style))
btn03.clicked.connect(lambda:Keypoints(fip1))

btn04 = QtWidgets.QPushButton(Form)    
btn04.setText('4.2 Matched Keypoints')      
btn04.setGeometry(100,450,200,50)
btn04.setStyleSheet((style))
btn04.clicked.connect(lambda:matchedKeypoints(fip1,fip2))

Form.show()
sys.exit(app.exec_())
