from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as T

def augmentation():
    batch_size = 1
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=T.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    transformTP = T.ToPILImage()
    orig_img = transformTP(images[0,:,:,:])

    plt.rcParams["savefig.bbox"] = 'tight'
    #orig_img = Image.open(images)
    # if you change the seed, make sure that the randomly-applied transforms
    # properly show that the image can be both transformed and *not* transformed!
    torch.manual_seed(0)

    nrows, ncols = 9, 5  # array of sub-plots
    figsize = [10, 10]     # figure size, inches
    _ , axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def plot(with_orig=True, row_title=None, **imshow_kwargs):
        for i in range(9):
            if i == 0:
                imgs = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
            elif i == 1:
                jitter = T.ColorJitter(brightness=.5, hue=.3)
                imgs = [jitter(orig_img) for _ in range(4)]
            elif i == 2:
                padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
                imgs = padded_imgs
            elif i == 3:
                affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
                imgs = [affine_transfomer(orig_img) for _ in range(4)]
            elif i == 4:
                applier = T.RandomApply(transforms=[T.RandomCrop(size=(20, 20))], p=0.5)
                imgs = [applier(orig_img) for _ in range(4)]
            elif i == 5:
                cropper = T.RandomCrop(size=(20, 20))
                imgs = [cropper(orig_img) for _ in range(4)]
            elif i == 6:
                hflipper = T.RandomHorizontalFlip(p=0.5)
                imgs = [hflipper(orig_img) for _ in range(4)]
            elif i == 7:
                perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
                imgs = [perspective_transformer(orig_img) for _ in range(4)]
            else:
                resize_cropper = T.RandomResizedCrop(size=(32, 32))
                imgs = [resize_cropper(orig_img) for _ in range(4)]

            if not isinstance(imgs[0], list):
                # Make a 2d grid even if there's just 1 row
                imgs = [imgs]
            
            for row_idx, row in enumerate(imgs):
                row = [orig_img] + row if with_orig else row
                for col_idx, img in enumerate(row):
                    ax = axs[i, col_idx]
                    ax.imshow(np.asarray(img), **imshow_kwargs)
                    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            if with_orig:
                axs[i, 0].set(title='Original image')
                axs[i, 0].title.set_size(8)
            '''
            if row_title is not None:
                for row_idx in range(num_rows):
                    axs[row_idx, 0].set(ylabel=row_title[row_idx])
            '''
        plt.tight_layout()

    '''
    padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
    plot(padded_imgs)

    center_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
    plot(center_crops)

    jitter = T.ColorJitter(brightness=.5, hue=.3)
    jitted_imgs = [jitter(orig_img) for _ in range(4)]
    plot(jitted_imgs)
    '''
    plot()
    plt.show()