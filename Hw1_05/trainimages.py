import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def trainimages():
    # GPU
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('GPU state:', device)

    '''torchvision.transforms.ToTensor
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    '''
    batch_size = 9

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)

    classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    nrows, ncols = 3, 3  # array of sub-plots
    figsize = [8, 8]     # figure size, inches
    _ , ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    '''
    for i in range(batch_size):
        pic = images[i,:,:,:].numpy()
        pic = np.transpose(pic,(1,2,0))
        plt.imshow(pic)
        plt.show()
    '''

    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        pic = images[i,:,:,:].numpy()
        pic = np.transpose(pic,(1,2,0))
        axi.axis('off')
        axi.imshow(pic)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title(classes[labels[i]])

    plt.tight_layout(True)
    plt.show()