import torch
from torchsummary import summary
import torchvision
import numpy as np

def model():
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU state:', device)
    model = torchvision.models.vgg19(weights=True,progress=True).to(device)
    print(model)
    summary(model, input_size=(3,224,224))