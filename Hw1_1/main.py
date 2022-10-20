from PyQt5              import QtWidgets, QtCore
import sys
from FindCorners        import FindCorners
from FindIntrinsic      import FindIntrinsic
from FindExtrinsic      import FindExtrinsic
from FindDistortion     import FindDistortion
from ShowResults        import ShowResults
from ShowWordsOn        import ShowWordsOn
from ShowWordsVertical  import ShowWordsVertical
from StereoDisparity    import StereoDisparity

fipL = " "
fipR = " "
FoP  = " "

#####Functions#####
def openFileL():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()  # 選取單一檔案
    global fipL
    fipL = str(filePath)
    print(filePath, filterType )

def openFileR():
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()  # 選取單一檔案
    global fipR
    fipR = str(filePath)    
    print(filePath, filterType )

def openFolder():
    folderPath = QtWidgets.QFileDialog.getExistingDirectory()       # 選取特定資料夾
    global FoP
    FoP = str(folderPath)
    print(folderPath)

def getCurrentNo(path):
    number = str(box.currentText())
    FindExtrinsic(path,number)

def getCurrentWordO(path):
    word = str(input.text())
    print (word)
    ShowWordsOn(path,word)

def getCurrentWordV(path):
    word = str(input.text())
    print (word)
    ShowWordsVertical(path,word)
    
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
Form.resize(1280, 800)                      # 設定長寬尺寸
Form.setStyleSheet('background:#BDB76B;')   # 使用網頁 CSS 樣式設定背景

#####Column 0#####

label0 = QtWidgets.QLabel(Form)
label0.setText('Load Image')
label0.setGeometry(50, 100, 250, 50)
label0.setAlignment(QtCore.Qt.AlignCenter)  # 對齊方式
label0.setWordWrap(True)
label0.setStyleSheet('''
    color:#FAEBD7;
    font-size:25px;
    font-weight:bold;
''')

btn01 = QtWidgets.QPushButton(Form)         # 在 Form 中加入一個 QPushButton
btn01.setText('Load Folder')                # 按鈕文字
btn01.setGeometry(100,200,200,50)           # 移動到 (100,200)，大小 200x50
btn01.setStyleSheet(style)
btn01.clicked.connect(openFolder)

btn02 = QtWidgets.QPushButton(Form)
btn02.setText('Load Image_L')   
btn02.setGeometry(100,400,200,50)
btn02.setStyleSheet((style))
btn02.clicked.connect(openFileL)

btn03 = QtWidgets.QPushButton(Form)    
btn03.setText('Load Image_R')      
btn03.setGeometry(100,600,200,50)
btn03.setStyleSheet((style))
btn03.clicked.connect(openFileR)

#####Column 1#####

label11 = QtWidgets.QLabel(Form)
label11.setText('1. Calibration')
label11.setGeometry(350, 100, 250, 50)
label11.setAlignment(QtCore.Qt.AlignCenter)
label11.setWordWrap(True)
label11.setStyleSheet('''
    color:#FAEBD7;
    font-size:25px;
    font-weight:bold;
''')

btn11 = QtWidgets.QPushButton(Form)
btn11.setText('1.1 Find Corners')
btn11.setGeometry(400,200,160,50)
btn11.setStyleSheet((style))
btn11.clicked.connect(lambda:FindCorners(FoP))

btn12 = QtWidgets.QPushButton(Form)
btn12.setText('1.2 Find Intrinsic')
btn12.setGeometry(400,300,160,50)
btn12.setStyleSheet((style))
btn12.clicked.connect(lambda:FindIntrinsic(FoP))

label12 = QtWidgets.QLabel(Form)
label12.setText('1.3 Find Extrinsic')
label12.setGeometry(350, 400, 150, 50)
label12.setAlignment(QtCore.Qt.AlignCenter)
label12.setWordWrap(True)
label12.setStyleSheet('''
    color:#FAEBD7;
    font-size:15px;
    font-weight:bold;
''')

box = QtWidgets.QComboBox(Form)
box.addItems(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
box.setGeometry(400,440,160,50)

btn13 = QtWidgets.QPushButton(Form)
btn13.setText('1.3 Find Extrinsic')
btn13.setGeometry(400,500,160,50)
btn13.setStyleSheet((style))
btn13.clicked.connect(lambda:getCurrentNo(FoP))

btn14 = QtWidgets.QPushButton(Form)
btn14.setText('1.4 Find Distortion')
btn14.setGeometry(400,600,160,50)
btn14.setStyleSheet((style))
btn14.clicked.connect(lambda:FindDistortion(FoP))

btn15 = QtWidgets.QPushButton(Form)
btn15.setText('1.5 Show Results')
btn15.setGeometry(400,700,160,50)
btn15.setStyleSheet((style))
btn15.clicked.connect(lambda:ShowResults(FoP))

#####Column 2#####

label2 = QtWidgets.QLabel(Form)
label2.setText('2. Augmented Reality')
label2.setGeometry(650, 100, 250, 50)
label2.setAlignment(QtCore.Qt.AlignCenter)
label2.setWordWrap(True)
label2.setStyleSheet('''
    color:#FAEBD7;
    font-size:25px;
    font-weight:bold;
''')

input = QtWidgets.QLineEdit(Form)   # 建立單行輸入框
input.setGeometry(700,200,200,50)   # 設定位置和尺寸
input.setMaxLength(6)


btn21 = QtWidgets.QPushButton(Form)
btn21.setText('2.1 Show Words on Board')
btn21.setGeometry(700,400,200,50)
btn21.setStyleSheet((style))
btn21.clicked.connect(lambda:getCurrentWordO(FoP))

btn22 = QtWidgets.QPushButton(Form)
btn22.setText('2.2 Show Words Vertically')
btn22.setGeometry(700,600,200,50)
btn22.setStyleSheet((style))
btn22.clicked.connect(lambda:getCurrentWordV(FoP))

#####Column 3#####

label3 = QtWidgets.QLabel(Form)
label3.setText('3. Stereo Disparity Map')
label3.setGeometry(950, 100, 300, 50)
label3.setAlignment(QtCore.Qt.AlignCenter)
label3.setWordWrap(True)
label3.setStyleSheet('''
    color:#FAEBD7;
    font-size:25px;
    font-weight:bold;
''')

btn31 = QtWidgets.QPushButton(Form)
btn31.setText('3.1 Stereo Disparity Map')
btn31.setGeometry(1000,400,200,50)
btn31.setStyleSheet((style))
btn31.clicked.connect(lambda:StereoDisparity(fipL,fipR))

Form.show()
sys.exit(app.exec_())
