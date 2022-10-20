import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.optim import SGD
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 30
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the CIFAR10 dataset
print("[INFO] loading the CIFAR10 dataset...")
transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
trainData = CIFAR10(root="./data", train=True,  download=True, transform=transform)
testData  = CIFAR10(root="./data", train=False, download=True, transform=transform)

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples      = int(len(trainData) * TRAIN_SPLIT)
numValSamples        = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples], generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader   = DataLoader(valData,  batch_size=BATCH_SIZE)
testDataLoader  = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps   = len(valDataLoader.dataset)   // BATCH_SIZE
testSteps  = len(testDataLoader.dataset)  // BATCH_SIZE


# initialize the VGG19 model
print("[INFO] initializing the VGG19 model...")
model = models.vgg19(weights = 'IMAGENET1K_V1')
input_lastLayer = model.classifier[6].in_features
model.classifier[6] = nn.Linear(input_lastLayer,10)
model = model.to(device)

# initialize our optimizer and loss function
opt = SGD(model.parameters(), lr = INIT_LR, momentum=0.9,weight_decay=5e-4)
lossFn = nn.CrossEntropyLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc" : [],
	"val_loss"  : [],
	"val_acc"   : [],
	"test_loss" : [],
	"test_acc"  : [],
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()


# loop over our epochs
for e in range(0, EPOCHS):
	# set the model in training mode
	model.train()

	# initialize the total training and validation and testing loss
	totalTrainLoss = 0
	totalValLoss   = 0
	totalTestLoss  = 0
	# initialize the number of correct predictions in the training
	# and validation step and testing step
	trainCorrect = 0
	valCorrect   = 0
	testCorrect  = 0
	# loop over the training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFn(pred, y)
			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
		
		# loop over the testing set
		for (x, y) in testDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions
			pred = model(x)
			totalTestLoss += lossFn(pred, y)
			# calculate the number of correct predictions
			testCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss   = totalValLoss   / valSteps
	avgTestLoss  = totalTestLoss  / testSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect   = valCorrect   / len(valDataLoader.dataset)
	testCorrect  = testCorrect	/ len(testDataLoader.dataset)

	# update our training history

	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	H["test_acc"].append(testCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss, valCorrect))
	print("Test loss: {:.6f}, Test accuracy: {:.4f}\n".format(avgTestLoss, testCorrect))


# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)

		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"],   label="val_loss")
plt.plot(H["test_loss"],  label="test_loss")
plt.plot(H["train_acc"],  label="train_acc")
plt.plot(H["val_acc"],    label="val_acc")
plt.plot(H["test_acc"],   label="test_acc")

plt.title("Loss and Accuracy on Dataset and Testset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
# serialize the model to disk
torch.save(model, args["model"])

# generate a classification report
print(classification_report(torch.tensor(testData.targets, device='cpu'), np.array(preds), target_names=testData.classes))