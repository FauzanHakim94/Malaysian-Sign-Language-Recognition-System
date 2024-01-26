import model_utils.ImportDataset as ImportDataset

import cv2
from PIL import Image
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import seaborn as sn
import pandas as pd

from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset management and creates mini batches

test_loader = 0

class TrainingNN:

    def startTrain(self, load_Model):
        global device, train_loader, test_loader
        rootPath = os.path.dirname(os.path.abspath(__file__))
        index = rootPath.find("66. Malaysia Sign Language Recognition System")
        rootPath = rootPath[:index + 45]

        torchDatasetPath = rootPath + '\\torchDataset\\training'
        csvPath = torchDatasetPath + "\\train.csv"
        traindataset = ImportDataset.CustomImageDataset(
            csv_file=csvPath,
            root_dir=torchDatasetPath,
            transform=transforms.ToTensor(),
        )
        torchDatasetPath = rootPath + '\\torchDataset\\test'
        csvPath = torchDatasetPath + "\\test.csv"
        testdataset = ImportDataset.CustomImageDataset(
            csv_file=csvPath,
            root_dir=torchDatasetPath,
            transform=transforms.ToTensor(),
        )
        """
        torchDatasetPath = rootPath + '\\landmarkDataset'
        csvPath = rootPath + '\\landmarkDataset' + "\Landmark Image Label.csv"

        dataset = ImportDataset.CustomImageDataset(
            csv_file=csvPath,
            root_dir=torchDatasetPath,
            transform=transforms.ToTensor(),
        )
        """

        # Hyperparameters
        in_channel = 3
        num_classes = 29
        learning_rate = 1e-3
        batch_size = 128
        num_epochs = 15
        model_Name = "MSLRV3.pth.tar"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #train_set, test_set = torch.utils.data.random_split(dataset, [1680, 719])
        train_loader = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=True)

        #model = torchvision.models.googlenet(pretrained=True)
        model = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(36864, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )


        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if load_Model:
            TrainingNN.load_checkpoint(torch.load(rootPath + "\\" + model_Name), model, optimizer)
        print("Create Confusion Matrix")
        TrainingNN.create_confusionMatrix(test_loader, model)
        print("Cofusion Matrix Created")
        # Train Network
        for epoch in range(num_epochs):
            losses = []

            if epoch % 4 == 0 and epoch > 1:
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                TrainingNN.save_checkpoint(checkpoint, filename="\\" + model_Name)
                TrainingNN.create_confusionMatrix(test_loader, model)

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)

                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()

            print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
            print("Checking accuracy on Training Set")
            TrainingNN.check_accuracy(train_loader, model)

            print("Checking accuracy on Test Set")
            TrainingNN.check_accuracy(test_loader, model)

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        TrainingNN.save_checkpoint(checkpoint, filename="\\" + model_Name)
        TrainingNN.create_confusionMatrix(test_loader, model)

    # Check accuracy on training to see how good our model is
    def check_accuracy(loader, model):
        global device
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                #print(x)
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )

        model.train()

    def getRocAC(loader, model):
        global device
        all_y_true = []
        all_y_pred = []

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                #print(x)
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                print("Ts1")
                all_y_true = np.append(all_y_true, y)
                print("Ts2")
                all_y_pred = np.append(all_y_pred, predictions)
                print("Ts3")

            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )

        y_true = all_y_true
        y_pred = all_y_pred

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], "k–")
        plt.plot(fpr, tpr, label="CNN(area = {:.3f})".format(roc_auc))
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="best")
        plt.show()

    def save_checkpoint(state, filename):
        rootPath = os.path.dirname(os.path.abspath(__file__))
        index = rootPath.find("66. Malaysia Sign Language Recognition System")
        rootPath = rootPath[:index + 45]

        print("=> Saving checkpoint")
        torch.save(state, rootPath + filename)

    def load_checkpoint(checkpoint, model, optimizer):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    def create_confusionMatrix(test_loader, model):
        y_pred = []
        y_true = []

        # iterate over test data
        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)  # Feed Network

                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)  # Save Prediction

                y = y.data.cpu().numpy()
                y_true.extend(y)  # Save Truth

        # constant for classes
        classes = ('A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Bapa','Emak','Melayu','Cina','India')

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 29, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('ConfusionMatrix.png')

    def pre_image(image_path, model, flag):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if flag:
            img = Image.open(image_path)
        else:
            img = image_path

        transform_norm = transforms.Compose([transforms.ToTensor()])
        # get normalized image
        img_normalized = transform_norm(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        # input = Variable(image_tensor)
        img_normalized = img_normalized.to(device)
        # print(img_normalized.shape)
        with torch.no_grad():
            model.eval()
            output = model(img_normalized)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, classes = torch.max(probs, 1)
            # print(output)
            index = output.data.cpu().numpy().argmax()
            return index, conf

#TrainingNN.startTrain(TrainingNN,1)

"""
rootPath = os.path.dirname(os.path.abspath(__file__))
index = rootPath.find("66. Malaysia Sign Language Recognition System")
rootPath = rootPath[:index + 45]
torchDatasetPath = rootPath + '\\torchDataset\\test'
csvPath = torchDatasetPath + "\\test.csv"
testdataset = ImportDataset.CustomImageDataset(
    csv_file=csvPath,
    root_dir=torchDatasetPath,
    transform=transforms.ToTensor(),
)
test_loader = DataLoader(dataset=testdataset, batch_size=128, shuffle=True)

global model, modelName
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rootPath = os.path.dirname(os.path.abspath(__file__))
index = rootPath.find("66. Malaysia Sign Language Recognition System")
rootPath = rootPath[:index + 45]

# model = torchvision.models.googlenet(pretrained=True)
model = nn.Sequential(

    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(36864, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 24)
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
TrainingNN.load_checkpoint(torch.load(rootPath + "\\" + "MSLRV3" + ".pth.tar"), model, optimizer)
print("Checking accuracy on Test Set")
#TrainingNN.getRocAC(test_loader, model)
"""


