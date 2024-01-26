import os
import shutil
import csv
#classes = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
classes = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Bapa', 'Emak', 'Melayu', 'Cina', 'India']
class PrepareImageDataset:
    def SplitDataset(self, imgPerGesture, train):
        test = 1-train

        rootPath = os.path.dirname(os.path.abspath(__file__))
        originDatasetPath = rootPath + "\landmarkDataset"
        targetDatasetPath = rootPath + "\\torchDataset"
        trainDatasetPath = targetDatasetPath+"\\training"
        trainLabelPath = trainDatasetPath + "\\train.csv"
        testDatasetPath = targetDatasetPath + "\\test"
        testLabelPath = testDatasetPath + "\\test.csv"

        if not os.path.exists(trainDatasetPath):
            os.makedirs(trainDatasetPath)
        if not os.path.exists(testDatasetPath):
            os.makedirs(testDatasetPath)
        if not os.path.exists(trainLabelPath):
            f = open(trainLabelPath, 'w')
            f.close()
        if not os.path.exists(testLabelPath):
            f = open(testLabelPath, 'w')
            f.close()

        trainRows = []
        testRows = []
        counter = 0
        for file in os.listdir(originDatasetPath):
            if file.endswith(".jpg"):

                counter = counter + 1
                filename = file
                labelIndex = 0
                numberIndex = 0
                for x in range(3):
                    filename = filename[1:]
                    imgLabelIndex = filename.find(" ")
                    filename = filename[imgLabelIndex:]
                    numberIndex = numberIndex+imgLabelIndex+1
                    if x == 1:
                        labelIndex = numberIndex

                imgLabelIndex2 = file.find(".jpg")
                imageLabel = file[labelIndex:numberIndex]
                #print(imageLabel)
                imageClass = 0

                for x in range(len(classes)):
                    print(classes[x])
                    print(imageLabel[1:])
                    if classes[x] == imageLabel[1:]:
                        imageClass = x
                        print("Found")
                imgLabelCounter = file[numberIndex:imgLabelIndex2]

                if int(imgLabelCounter) <= train*imgPerGesture/2:
                    print("copy:" + file[labelIndex:] )
                    trainRows.append([file, imageClass])
                    shutil.copy(originDatasetPath + "\\" + file, trainDatasetPath + "\\" + file)
                elif int(imgLabelCounter) > imgPerGesture/2+test*imgPerGesture/2:
                    print("copy:" + file[labelIndex:] )
                    trainRows.append([file, imageClass])
                    shutil.copy(originDatasetPath + "\\" + file, trainDatasetPath + "\\" + file)
                else:
                    print("not copy:" + file[labelIndex:] )
                    testRows.append([file, imageClass])
                    shutil.copy(originDatasetPath + "\\" + file, testDatasetPath + "\\" + file)

                if counter >= imgPerGesture:
                    counter = 0

        print(trainRows)
        print(testRows)
        csvfile = open(trainLabelPath, 'w', newline='')
        csvfile.truncate()
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(trainRows)
        csvfile.close()
        csvfile = open(testLabelPath, 'w', newline='')
        csvfile.truncate()
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(testRows)
        csvfile.close()

    def checkDataset(self):
        rootPath = os.path.dirname(os.path.abspath(__file__))
        targetDatasetPath = rootPath + "\\torchDataset"
        trainDatasetPath = targetDatasetPath+"\\training"
        trainLabelPath = trainDatasetPath + "\\train.csv"
        testDatasetPath = targetDatasetPath + "\\test"
        testLabelPath = testDatasetPath + "\\test.csv"

        csvfile = open(trainLabelPath, 'r')
        csvreader = csv.reader(csvfile)
        trainRow = []
        for row in csvreader:
            trainRow.append(row)
        csvfile.close()
        csvfile = open(testLabelPath, 'r')
        csvreader = csv.reader(csvfile)
        testRow = []
        for row in csvreader:
            testRow.append(row)
        csvfile.close()

        trainFileStatus = []
        testFileStatus = []
        for file in os.listdir(trainDatasetPath):
            if file.endswith(".jpg"):
                for x in range(len(trainRow)):
                    rowIndex = trainRow[x]
                    if rowIndex[0] == file:
                        trainFileStatus.append("Found")
        for file in os.listdir(testDatasetPath):
            if file.endswith(".jpg"):
                for x in range(len(testRow)):
                    rowIndex = testRow[x]
                    if rowIndex[0] == file:
                        testFileStatus.append("Found")

        print(trainFileStatus)
        print(len(trainFileStatus))
        print(testFileStatus)
        print(len(testFileStatus))


PrepareImageDataset.SplitDataset(PrepareImageDataset, 100, 0.7)
PrepareImageDataset.checkDataset(PrepareImageDataset)