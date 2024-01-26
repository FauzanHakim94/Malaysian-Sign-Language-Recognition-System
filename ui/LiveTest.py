import model_utils.TrainModel as TrainModel
import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
from skimage import io
from torch.utils.data import Dataset
from skimage import color
from PIL import Image
from PIL import ImageDraw
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy

rootPathRecord = "null"
alphabetsRecord = "null"
model = 0
modelName = 'null'

mpHolistic = 0
mpDraws = 0
mpDrawingStyles = 0
holistic = 0

cameraButtonState = 0
cameraID = 0

savedC = 0

class LTab:
    def init(self, rootPath, datasetPath, alphabets):
        global rootPathRecord, alphabetsRecord, mpHolistic, mpDraws, mpDrawingStyles, holistic, modelName, tempPath, mpFaceMesh, faceMesh
        rootPathRecord = datasetPath
        tempPath = rootPath+"\\temp"
        alphabetsRecord = alphabets

        ###Message Box
        LTab.msg = QMessageBox()

        LTab.simulateTab = self.tabs.findChild(QWidget, "SIMULATETab")
        LTab.liveCameraButton = LTab.simulateTab.findChild(QPushButton, "liveCameraButton")
        LTab.modelCombo = LTab.simulateTab.findChild(QComboBox, "modelCombo")
        LTab.RoIWindow = LTab.simulateTab.findChild(QLabel, "RoIWindow")
        LTab.PredictedClassLabel = LTab.simulateTab.findChild(QLabel, "PredictedClassLabel")
        #LTab.ConfidenceLabel = LTab.simulateTab.findChild(QLabel, "ConfidenceLabel")
        LTab.liveScreen = LTab.simulateTab.findChild(QLabel, "liveScreen")

        LTab.modelCombo.addItem("MSLR")
        LTab.modelCombo.addItem("MSLRV2")
        LTab.modelCombo.addItem("MSLRV3")
        modelName = LTab.modelCombo.currentText()

        #widget connection
        LTab.liveCameraButton.clicked.connect(LTab.liveCameraButtonClicker)
        LTab.modelCombo.currentIndexChanged.connect(LTab.modelOnChanged)

        LTab.RoIWindow.setText("Camera is OFF")
        LTab.PredictedClassLabel.clear()
        #LTab.ConfidenceLabel.clear()

        LTab.liveScreen.setText("Camera is OFF")
        LTab.liveScreen.setMaximumWidth(1400)
        LTab.liveScreen.setMaximumHeight(500)

        mpHolistic = mp.solutions.holistic
        mpDraws = mp.solutions.drawing_utils
        mpDrawingStyles = mp.solutions.drawing_styles
        holistic = mpHolistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85)

        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5)


        ##Initial Activity
        LTab.Worker3 = Worker3()
        LTab.Worker3.ImageUpdate.connect(LTab.ImageUpdateSlot)
        LTab.Worker3.ImageRoI.connect(LTab.RoIUpdateSlot)

    def callMessageBox(self, title, message):
        LTab.msg.setWindowTitle(title)
        LTab.msg.setText(message)
        LTab.msg.setIcon(QMessageBox.Critical)
        LTab.msg.setStandardButtons(QMessageBox.Close)
        LTab.msg.exec_()

    def liveCameraButtonClicker(self):
        global cameraButtonState
        cameraButtonState = not cameraButtonState
        LTab.threadState(self, cameraButtonState)

    def modelOnChanged(self):
        global modelName, cameraButtonState
        modelName = LTab.modelCombo.currentText()
        cameraButtonState = 0
        LTab.threadState(self, cameraButtonState)
        #self.qlabel.setText(text)
        #self.qlabel.adjustSize()

    def threadState(self, state):
        if state:
            if not LTab.Worker3.isRunning():
                LTab.Worker3.start()
                LTab.liveScreen.clear()
                LTab.liveScreen.setText("Turning ON camera. Please wait...")
        else:
            if LTab.Worker3.isRunning():
                LTab.Worker3.stop()
                LTab.liveScreen.clear()
                LTab.liveScreen.setText("Camera is OFF")

    #update image for webcam
    def ImageUpdateSlot(FlippedImage, buff_PredictedClassLabelY, buff_ConfidenceLabelY, buff_liveTextY, liveTextCounter_IntY):
        global savedC, tempPath
        LTab.PredictedClassLabel.setText(buff_PredictedClassLabelY)
        #LTab.ConfidenceLabel.setText(buff_ConfidenceLabelY)

        liveTextCounter_IntY = liveTextCounter_IntY*10
        print(liveTextCounter_IntY)

        shapes = np.zeros_like(FlippedImage, np.uint8)
        cv2.rectangle(shapes, (20, 380), (620, 440), (255, 255, 255), cv2.FILLED)
        out = FlippedImage.copy()
        alpha = 0.25
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(FlippedImage, alpha, shapes, 1 - alpha, 0)[mask]
        FlippedImage = out

        cv2.putText(img=FlippedImage, text=buff_liveTextY, org=(320-liveTextCounter_IntY, 420), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(0, 0, 0), thickness=2)

        #cv2.imwrite(tempPath + "\\liveText.png", FlippedImage)
        #FlippedImage = cv2.imread(tempPath + "\\liveText.png")
        ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0],
                                   QImage.Format_RGB888)
        FlippedImage = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

        LTab.liveScreen.clear()
        LTab.liveScreen.setPixmap(QPixmap.fromImage(FlippedImage))

    def RoIUpdateSlot(image):
        LTab.RoIWindow.clear()
        LTab.RoIWindow.setPixmap(QPixmap.fromImage(image))

liveTextCounterX = 0
liveTextWordCounterX = 0
liveTextBufferX = " "
prevPredictedClassX = " "
buff_PredictedClassLabel, buff_ConfidenceLabel, buff_liveText = " ", " ", " "
class Worker3(QThread):
    global cameraID, cameraButtonState, mpHolistic, mpDraws, mpDrawingStyles, holistic, modelName, tempPath, mpFaceMesh, faceMesh

    ImageUpdate = pyqtSignal(numpy.ndarray, str, str, str, int)
    ImageRoI = pyqtSignal(QImage)

    def run(self):
        self.loadModel()
        self.Capture = cv2.VideoCapture(cameraID)
        self.ThreadActive = True

        while self.ThreadActive:
            print("Start")
            ret, frame = self.Capture.read()
            takenImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            handResult = 0
            h, w, c = takenImage.shape
            results = holistic.process(takenImage)
            print("Sini")
            if results.right_hand_landmarks:
                handResult = results.right_hand_landmarks
            if results.left_hand_landmarks:
                handResult = results.left_hand_landmarks
            if handResult:
                x_min, x_max, y_min, y_max = self.drawBox(h, w, handResult)
                cv2.rectangle(takenImage, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

                if x_min>0 and x_max<w and y_min>0 and y_max<h:
                    # create empty canvas and retrace image
                    canvas = np.zeros((x_max - x_min, y_max - y_min, 3), np.uint8)
                    print("Test")
                    if results.face_landmarks:
                        for landmarkIndex in range(len(results.face_landmarks.landmark)):
                            results.face_landmarks.landmark[landmarkIndex].x = (((results.face_landmarks.landmark[
                                                                                      landmarkIndex].x) / 1 * 640) - x_min) / (
                                                                                           x_max - x_min)
                            results.face_landmarks.landmark[landmarkIndex].y = (((results.face_landmarks.landmark[
                                                                                      landmarkIndex].y) / 1 * 480) - y_min) / (
                                                                                           y_max - y_min)

                        mpDraws.draw_landmarks(
                            image=canvas,
                            landmark_list=results.face_landmarks,
                            connections=mpFaceMesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mpDrawingStyles
                                .get_default_face_mesh_contours_style())

                    print("Test2")
                    for landmarkIndex in range(21):
                        handResult.landmark[landmarkIndex].x = (((handResult.landmark[
                                                                      landmarkIndex].x) / 1 * 640) - x_min) / (
                                                                       x_max - x_min)
                        handResult.landmark[landmarkIndex].y = (((handResult.landmark[
                                                                      landmarkIndex].y) / 1 * 480) - y_min) / (
                                                                       y_max - y_min)
                    mpDraws.draw_landmarks(canvas, handResult, mpHolistic.HAND_CONNECTIONS,
                                           mpDrawingStyles.get_default_hand_landmarks_style(),
                                           mpDrawingStyles.get_default_hand_connections_style())
                    canvas = cv2.resize(canvas, (100, 100))
                    canvas = cv2.flip(canvas, 1)

                    cv2.imwrite(tempPath+"\\temp.png", canvas)
                    im_pil = io.imread(tempPath+"\\temp.png")
                    im_pil = Image.fromarray(im_pil)
                    imageClass, conf = self.predictImageClass(im_pil)
                    im_pil = self.convert_cv_qt(0, canvas, 100, 100)
                    self.ImageRoI.emit(im_pil)
                    confidence = str(conf)
                    ce = confidence.find('tensor')
                    confidence = confidence[ce+8:-2]
                    print("Sini")
                    if conf>0.75:
                        buff_PredictedClassLabelX, buff_ConfidenceLabelX, buff_liveTextX, liveTextCounter_IntX = self.updateLiveText(imageClass[8:], confidence, 1)
                        #self.ImageClass.emit(imageClass[8:], confidence, 1)
                    else:
                        buff_PredictedClassLabelX, buff_ConfidenceLabelX, buff_liveTextX, liveTextCounter_IntX = self.updateLiveText('none', 'none', 0)
                        #self.ImageClass.emit('none', 'none', 0)
            else:
                buff_PredictedClassLabelX, buff_ConfidenceLabelX, buff_liveTextX, liveTextCounter_IntX = self.updateLiveText(' ', 'none', 1)

            #Process Image to add live text
            FlippedImage = cv2.flip(takenImage, 1)

            if ret:
                if self.ThreadActive:
                    self.ImageUpdate.emit(FlippedImage, buff_PredictedClassLabelX, buff_ConfidenceLabelX, buff_liveTextX, liveTextCounter_IntX)
                    if not handResult:
                        canvas = np.zeros((100, 100, 3), np.uint8)
                        canvas = cv2.resize(canvas, (100, 100))
                        im_pil = self.convert_cv_qt(0, canvas, 100, 100)
                        self.ImageRoI.emit(im_pil)

    def stop(self):
        self.ThreadActive = False
        self.Capture.release()
        self.terminate()

    def updateLiveText(self, imageClass, confidence, flag):
        global liveTextCounterX, liveTextWordCounterX, liveTextBufferX, prevPredictedClassX, buff_PredictedClassLabel, buff_ConfidenceLabel, buff_liveText
        dummyBuff = " "
        imgClass = imageClass
        # if there is prediction
        if flag:
            # Update Predicted Gesture Label
            if not imageClass == " ":
                buff_PredictedClassLabel = "Gesture " + imageClass
                confidenceLevel = '%.2f' % (float(confidence) * 100)
                buff_ConfidenceLabel = confidenceLevel + "%"
            # If the alphabet repeat
            if prevPredictedClassX == imgClass:
                liveTextCounterX += 1
            else:
                liveTextCounterX = 0
                prevPredictedClassX = imgClass
            # if the alphabet repeat X times
            if liveTextCounterX > 5:
                liveTextCounterX = 0
                # Append the alphabet to array
                liveTextBufferX = liveTextBufferX + imgClass
                if len(liveTextBufferX) >= 25:
                    liveTextBufferX = liveTextBufferX[1:]
                if len(imageClass)>1:
                    print("panjang: " +str(len(imageClass)))
                    liveTextWordCounterX += len(imageClass)
                else:
                    liveTextWordCounterX += 1
            # display the array
            if not liveTextCounterX == 0:
                buff_liveText = liveTextBufferX
            if liveTextWordCounterX >= 25:
                liveTextWordCounterX = 25

        else:
            buff_PredictedClassLabel = " "
            buff_ConfidenceLabel = " "

        buffLength = len(liveTextBufferX)
        for b in range(buffLength-1):
            dummyBuff = dummyBuff + " "
        if dummyBuff == liveTextBufferX:
            liveTextWordCounterX = 0
            liveTextBufferX = " "

        liveTextCounter_Int = liveTextWordCounterX
        print("Counter: "+str(liveTextCounter_Int))
        return buff_PredictedClassLabel, buff_ConfidenceLabel, buff_liveText, liveTextCounter_Int

    def convert_cv_qt(self, flag, rgb_image, scaledWidth, scaledHeight):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #p = convert_to_Qt_format.scaled(scaledWidth, scaledHeight, Qt.KeepAspectRatio)
        return convert_to_Qt_format

    def drawBox(self, h, w, hand_landmarks):

        x_max = 0
        y_max = 0
        x_min = w
        y_min = h

        for c in range(21):
            x = int(hand_landmarks.landmark[c].x * w)
            y = int(hand_landmarks.landmark[c].y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        return x_min-15, x_max+15, y_min-15, y_max+15

    def loadModel(self):
        global model, modelName
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rootPath = os.path.dirname(os.path.abspath(__file__))
        index = rootPath.find("66. Malaysia Sign Language Recognition System")
        rootPath = rootPath[:index + 45]

        if modelName == "MSLRV3":
            classesNo = 29
        else:
            classesNo = 24

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
            nn.Linear(512, classesNo)
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr= 1e-3)
        TrainModel.TrainingNN.load_checkpoint(torch.load(rootPath + "\\" + modelName + ".pth.tar"), model, optimizer)

    def predictImageClass(self, frame=None):
        global model
        predict_class, conf = TrainModel.TrainingNN.pre_image(frame,model,0)
        gestureClass = 'Gesture '+alphabetsRecord[predict_class]
        return gestureClass, conf
