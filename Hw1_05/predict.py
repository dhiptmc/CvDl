from PyQt5       import QtWidgets
import torch
from torch import nn
from PIL import Image
from torchvision import transforms,datasets,models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def predict(path):
    def show():
        Form = QtWidgets.QWidget()
        mbox = QtWidgets.QMessageBox(Form)
        mbox.critical(Form, 'critical', 'No File selected! Please load image!')
    if path == " ":
        show()
    else:
        classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''
        model = models.vgg19(weights = 'IMAGENET1K_V1')
        input_lastLayer = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_lastLayer,10)
        '''
        model = torch.load("./model/vgg19_v3.pth")
        model = model.to(device)
        model.eval()

        img_path = path
        
        transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        img = Image.open(img_path)
        imgT = transform(img).unsqueeze(0)

        imgT = imgT.to(device)
        outputs = model(imgT)

        _ , indices = torch.max(outputs,1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        confidenceO = torch.nn.functional.softmax(outputs, dim=1)[0]
        perc = percentage[int(indices)].item()
        confidence = confidenceO[int(indices)].item()
        result = classes[indices]
        print('predicted:', result, perc)

        showimg = mpimg.imread(path)
        imgplot = plt.imshow(showimg)
        plt.title("Confidence = %f \n Prediciton Label: %s" %(confidence, result))
        plt.tight_layout(True)
        plt.show()