import os
import sys
import cv2
import time

import mediapipe as mp
from openpyxl import Workbook, load_workbook
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

rootPathRecord = "null"
datasetPathRecord = "null"
alphabetsRecord = "null"
headerFields = ['File Name', 'Classes', 'Width', 'Height', 'X Min', 'X Max', 'Y Min', 'Y Max']
cameraButtonState = 0
cameraID = 0
listRow = 0
frameCounter = 0
displayModelX = 0
imgNo = 0
startTimeModel = 0
snapButtonState = 0
startTime = 0
mpHolistic = 0
mpDraws = 0
mpDrawingStyles = 0
holistic = 0
excelFlag = 0
wb = 0
handCount = 0

class VTab:
    def init(self, rootPath, alphabets, datasetPath):
        global rootPathRecord, datasetPathRecord, alphabetsRecord, mpHolistic, mpDraws, holistic, mpDrawingStyles, mpFaceMesh, faceMesh
        rootPathRecord = rootPath
        datasetPathRecord = datasetPath
        alphabetsRecord = alphabets

        # Message Box
        VTab.msg = QMessageBox()
        VTab.searchLineEdit = self.findChild(QLineEdit, "filePathEditLine")

        VTab.recordTab = self.tabs.findChild(QWidget, "RECORDTab")
        VTab.list1 = VTab.recordTab.findChild(QListWidget, "listWidget1")
        VTab.scrollbar1 = VTab.recordTab.findChild(QScrollBar, "vScrollBar1")
        VTab.cameraButton = VTab.recordTab.findChild(QPushButton, "cameraButton")
        VTab.spinCameraID = VTab.recordTab.findChild(QSpinBox, "spinBoxCamera")
        VTab.snapButton = VTab.recordTab.findChild(QPushButton, "snapButton")
        VTab.liveImage = VTab.recordTab.findChild(QLabel, "liveImage")
        VTab.labelCaptureProgress = VTab.recordTab.findChild(QLabel, "labelCaptureProgress")
        VTab.capturedProgress = VTab.recordTab.findChild(QProgressBar, "capturedProgress")

        # widget connection
        VTab.cameraButton.clicked.connect(VTab.cameraButtonClicker)
        VTab.spinCameraID.valueChanged.connect(VTab.cameraIDChanged)
        VTab.snapButton.clicked.connect(VTab.snapButtonClicker)
        VTab.list1.itemClicked.connect(VTab.itemClickedEvent)

        # define item in List
        VTab.addList(self, alphabets);

        # display initial image
        VTab.liveImage.setText("Camera is OFF")
        VTab.liveImage.setMaximumWidth(1400)
        VTab.liveImage.setMaximumHeight(500)

        # define framework
        mpHolistic = mp.solutions.holistic
        mpDraws = mp.solutions.drawing_utils
        mpDrawingStyles = mp.solutions.drawing_styles
        holistic = mpHolistic.Holistic(min_detection_confidence=0.85, min_tracking_confidence=0.85)

        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5)

        # display initial tab progress
        VTab.labelCaptureProgress.setStyleSheet("background-color: white")
        VTab.labelCaptureProgress.clear()
        VTab.capturedProgress.setValue(0)

        # Initial Activity
        VTab.Worker1 = Worker1()
        VTab.Worker1.ImageUpdate.connect(VTab.ImageUpdateSlot)
        VTab.Worker1.ImageUpdate2.connect(VTab.ImageUpdate2Slot)
        VTab.Worker1.ProgressUpdate.connect(VTab.ProgressUpdateSlot)
        VTab.Worker1.NotDetect.connect(VTab.NotDetectSlot)

    def callMessageBox(self, title, message):
        VTab.msg.setWindowTitle(title)
        VTab.msg.setText(message)
        VTab.msg.setIcon(QMessageBox.Critical)
        VTab.msg.setStandardButtons(QMessageBox.Close)
        VTab.msg.exec_()

    """when on/off camera button clicked"""
    def cameraButtonClicker(self):
        global cameraButtonState
        #toggle camera button state
        cameraButtonState = not cameraButtonState
        VTab.threadState(self, cameraButtonState)

    """when snap camera button clicked"""
    def snapButtonClicker(self):
        global cameraButtonState, datasetPathRecord, snapButtonState, frameCounter, displayModelX,\
            startTimeModel, headerFields, wb, handCount

        #camera is not turned on
        if not cameraButtonState:
            VTab.callMessageBox(self, "Camera is not turned ON", "Please turn ON the camera first")
            return
        index = datasetPathRecord.find("\Gesture")

        #gesture is not selected
        if index<0:
            VTab.callMessageBox(self, "Gesture is not selected", "Please select any of gesture listed")
            return
        #finding dataset file for recording
        tempDir = datasetPathRecord[:index] + "\Record Image Label.xlsx"
        if os.path.exists(tempDir):
            wb = load_workbook(tempDir)
        else:
            workbook = Workbook()
            sheet = workbook.active
            workbook.save(filename=tempDir)
            wb = load_workbook(tempDir)
        sheet = wb.worksheets[0]
        #naming column title in spreadsheet
        for col in range(len(headerFields)):
            sheet.cell(row=1, column=col + 1).value = headerFields[col]
        wb.save(tempDir)
        #initial setup for thread
        snapButtonState = 1
        frameCounter = 0
        displayModelX = 1
        handCount = 1
        #initial setup for progress bar
        VTab.capturedProgress.setRange(0, 5)
        VTab.capturedProgress.setValue(0)
        #initial time for model image
        startTimeModel = time.time()

    """when changing camera channel"""
    def cameraIDChanged(self):
        global cameraID, cameraButtonState

        #initial setup when camera channel is changed
        VTab.liveImage.setText("Camera is OFF")
        cameraID = VTab.spinCameraID.value()
        cameraButtonState = 0
        VTab.threadState(self, cameraButtonState)

    """adding gesture type in the list"""
    def addList(self, items):
        for x in range(len(items)):
            VTab.list1.addItem(QListWidgetItem("Gesture "+items[x]))
        VTab.list1.setVerticalScrollBar(VTab.scrollbar1)

    """when gesture type is selected"""
    def itemClickedEvent(self):
        global listRow, datasetPathRecord, alphabetsRecord

        #update GUI search path
        listRow = VTab.list1.currentRow()
        index = datasetPathRecord.find("dataset")
        datasetPathRecord = datasetPathRecord[:index+7]+'\Gesture '+alphabetsRecord[listRow]
        VTab.searchLineEdit.setText(datasetPathRecord)

    """when threadState is updated"""
    def threadState(self, state):
        if state:
            # start thread if not running
            if not VTab.Worker1.isRunning():
                VTab.Worker1.start()
                VTab.liveImage.clear()
                VTab.liveImage.setText("Turning ON camera. Please wait...")
        else:
            # stop thread if running
            if VTab.Worker1.isRunning():
                VTab.Worker1.stop()
                VTab.liveImage.clear()
                VTab.liveImage.setText("Camera is OFF")

    """update image for webcam"""
    def ImageUpdateSlot(image):
        VTab.liveImage.clear()
        VTab.liveImage.setPixmap(QPixmap.fromImage(image))

    """update text on webcam image"""
    def ImageUpdate2Slot(string):
        VTab.liveImage.clear()
        VTab.liveImage.setText(string)

    """update progress bar"""
    def ProgressUpdateSlot(progress):
        if progress<=VTab.capturedProgress.maximum():
            VTab.capturedProgress.setValue(progress)
        else:
            VTab.capturedProgress.setValue(0)

    """feedback on user condition, no hand detect/ standing too close or far"""
    def NotDetectSlot(type, flag, progress):
        global handCount
        if type:
            if flag == 0:
                VTab.labelCaptureProgress.setText('You are too close... Please step backward')
            elif flag == 1:
                VTab.labelCaptureProgress.setText('You are too far... Please step forward')
            else:
                VTab.labelCaptureProgress.clear()
        else:
            if flag:
                handSide = ['Right', 'Left']
                VTab.labelCaptureProgress.setText('Your ' + handSide[handCount-1] + ' HAND is not detected')
            else:
                if progress < VTab.capturedProgress.maximum() - 1:
                    VTab.labelCaptureProgress.setText('Capturing images...')
                else:
                    VTab.labelCaptureProgress.clear()

"""Thread 1"""
class Worker1(QThread):
    global cameraID, cameraButtonState, frameCounter

    #slot to update
    ImageUpdate = pyqtSignal(QImage)
    ImageUpdate2 = pyqtSignal(str)
    ProgressUpdate = pyqtSignal(int)
    NotDetect = pyqtSignal(int, int, int)

    """when thread is running"""
    def run(self):
        # record image from webcam
        self.Capture = cv2.VideoCapture(cameraID)
        if self.Capture is None or not self.Capture.isOpened():
            cameraButtonState = 0
            self.ImageUpdate2.emit("Camera not detected in this channel. Please change the camera channel.")
            self.ThreadActive = False
        else:
            self.ThreadActive = True
            frameCounter = 0

        # if camera is detected and open
        while self.ThreadActive:
            global mpFaceMesh, faceMesh, rootPathRecord, alphabetsRecord, snapButtonState, startTime, imgNo,\
                displayModelX, startTimeModel, listRow, holistic, mpDraws, mpDrawingStyles, mpHolistic, wb, excelFlag, handCount, datasetPathRecord

            ##display model to mimic first
            #initally displayModelX is 1
            if displayModelX:
                #finding directory for model
                directory = datasetPathRecord
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if os.getcwd != directory:
                    os.chdir(directory)
                imgNo = len(os.listdir())
                index = rootPathRecord.find("66. Malaysia Sign Language Recognition System")
                #initally handCount is 1
                if handCount == 1:
                    hand = 'R'
                if handCount == 2:
                    hand = 'L'
                #display model
                if listRow<24:
                    model_directory = cv2.imread(rootPathRecord[:index+45]+'/supportGUI' + '/Gesture ' + \
                                      alphabetsRecord[listRow] + ' Model ' + hand + '.PNG')
                else:
                    model_directory = cv2.imread(rootPathRecord[:index+45]+'/supportGUI'+'/Gesture ' + \
                                                 alphabetsRecord[listRow]+'.PNG')

                if time.time() - startTimeModel > 5:
                    displayModelX = 0
                    startTime = time.time()
                else:
                    seconds = str(5-round(time.time() - startTimeModel))
                    cv2.putText(img=model_directory, text=seconds, org=(1155, 708), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                                color=(0, 0, 255), thickness=3)
                    model_directory = self.convert_cv_qt(model_directory)
                    self.ImageUpdate.emit(model_directory)
            ## start to record volunteer
            else:
                #record frame from video
                ret, frame = self.Capture.read()
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = Image.shape
                handResult = 0
                #process image using mediapipe holistic
                results = holistic.process(Image)
                if results.face_landmarks:
                    mpDraws.draw_landmarks(
                        image=Image,
                        landmark_list=results.face_landmarks,
                        connections=mpFaceMesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mpDrawingStyles
                            .get_default_face_mesh_contours_style())

                if handCount == 1 or handCount == 0:
                    handResult = results.right_hand_landmarks
                if handCount == 2:
                    handResult = results.left_hand_landmarks
                #if hand is successfully processed
                if handResult:
                    #draw handlandmark on video
                    x_min, x_max, y_min, y_max = self.drawBox(h, w, handResult)
                    cropped_image = frame[y_min:y_max, x_min:x_max]
                    cropped_image = cv2.flip(cropped_image, 1)

                    cv2.rectangle(Image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                    mpDraws.draw_landmarks(Image, handResult, mpHolistic.HAND_CONNECTIONS,
                                           mpDrawingStyles.get_default_hand_landmarks_style(),
                                           mpDrawingStyles.get_default_hand_connections_style())

                #if hand is not detected, display original video
                FlippedImage = cv2.flip(Image, 1)
                ## Start to record images if snap button is pressed
                # only execute if snap button has been pressed
                if snapButtonState:
                    if not displayModelX:
                        #snap image every 500ms
                        if time.time()-startTime > 1:
                            #if hand is detected
                            if handResult:
                                #increase frame counter
                                frameCounter = frameCounter + 1
                                self.NotDetect.emit(0, 0, frameCounter)
                                self.ProgressUpdate.emit(frameCounter)
                                #record image in spreadsheet
                                handSide = ['1Right', '2Left']
                                filename = '\Gesture ' + handSide[handCount - 1] + ' ' + alphabetsRecord[
                                    listRow] + ' ' + str(frameCounter + imgNo) + '.jpg'
                                ## record for dataset
                                index = rootPathRecord.find("66. Malaysia Sign Language Recognition System")
                                tempDir = rootPathRecord[:index + 45] + "\dataset" + "\Record Image Label.xlsx"
                                wb = load_workbook(tempDir)
                                excelFlag = 1
                                sheet = wb.worksheets[0]
                                lastRow = sheet.max_row
                                sheet.cell(row=lastRow + 1, column=1).value = 'hl'+str(lastRow)+filename[1:]
                                sheet.cell(row=lastRow + 1, column=2).value = listRow
                                sheet.cell(row=lastRow + 1, column=3).value = w
                                sheet.cell(row=lastRow + 1, column=4).value = h
                                sheet.cell(row=lastRow + 1, column=5).value = x_min
                                sheet.cell(row=lastRow + 1, column=6).value = x_max
                                sheet.cell(row=lastRow + 1, column=7).value = y_min
                                sheet.cell(row=lastRow + 1, column=8).value = y_max
                                wb.save(tempDir)
                                excelFlag = 0
                                ## record for landmarkDataset
                                tempDir = rootPathRecord[:index + 45] + "\landmarkDataset" + "\Landmark Image Label.xlsx"
                                wb = load_workbook(tempDir)
                                excelFlag = 1
                                sheet = wb.worksheets[0]
                                lastRow = sheet.max_row
                                sheet.cell(row=lastRow + 1, column=1).value = 'hl'+str(lastRow)+filename[1:]
                                sheet.cell(row=lastRow + 1, column=2).value = listRow
                                wb.save(tempDir)
                                excelFlag = 0
                                #create empty canvas and retrace image
                                canvas = np.zeros((x_max - x_min, y_max - y_min, 3), np.uint8)
                                landmarkIndex = 0
                                if results.face_landmarks:
                                    for landmarkIndex in range(len(results.face_landmarks.landmark)):
                                        results.face_landmarks.landmark[landmarkIndex].x = (((results.face_landmarks.landmark[landmarkIndex].x) / 1 * 640) - x_min) / (x_max - x_min)
                                        results.face_landmarks.landmark[landmarkIndex].y = (((results.face_landmarks.landmark[landmarkIndex].y) / 1 * 480) - y_min) / (y_max - y_min)

                                    mpDraws.draw_landmarks(
                                        image=canvas,
                                        landmark_list=results.face_landmarks,
                                        connections=mpFaceMesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mpDrawingStyles
                                            .get_default_face_mesh_contours_style())

                                for landmarkIndex in range(21):
                                    handResult.landmark[landmarkIndex].x = (((handResult.landmark[landmarkIndex].x) / 1 * 640) - x_min) / (x_max - x_min)
                                    handResult.landmark[landmarkIndex].y = (((handResult.landmark[landmarkIndex].y) / 1 * 480) - y_min) / (y_max - y_min)
                                mpDraws.draw_landmarks(canvas, handResult, mpHolistic.HAND_CONNECTIONS, mpDrawingStyles.get_default_hand_landmarks_style(), mpDrawingStyles.get_default_hand_connections_style())
                                canvas = cv2.resize(canvas, (100, 100))
                                #save original image in dataset
                                writePath = rootPathRecord[:index + 45] + "\dataset" + "\Gesture "+alphabetsRecord[listRow]
                                cropped_image = cv2.resize(cropped_image, (100, 100))
                                cv2.imwrite(writePath + '\\' + 'hl'+str(lastRow)+filename[1:], cropped_image)
                                #save retraced image in landmark dataset
                                writePath = rootPathRecord[:index + 45] + "\landmarkDataset"
                                #canvas = cv2.flip(canvas, 1)
                                cv2.imwrite(writePath + '\\' + 'hl'+str(lastRow)+filename[1:], canvas)
                                startTime = time.time()

                            # if hand is not detected during recording process
                            else:
                                self.NotDetect.emit(0, 1, frameCounter)
                        #if 10 images has been captured for 1 hand
                        if frameCounter>4:
                            #change to another hand
                            if handCount == 1:
                                snapButtonState = 1
                                frameCounter = 0
                                displayModelX = 1
                                handCount = 2
                                startTimeModel = time.time()
                            #finish capturing process
                            elif handCount == 2:
                                handCount = 0
                                frameCounter = 0
                                snapButtonState = 0

                            else:
                                print('nothing')
                            self.ProgressUpdate.emit(frameCounter)
                            #snapButtonState = 0

                #displaying image
                if ret:
                    if snapButtonState:
                        if not displayModelX:
                            cv2.putText(img=FlippedImage, text="Captured: "+str(frameCounter+1), org=(10,20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                                    color=(0, 0, 255), thickness=1)
                    ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0],
                                               QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    if self.ThreadActive:
                        self.ImageUpdate.emit(Pic)
    """stop the thread"""
    def stop(self):
        self.ThreadActive = False
        self.Capture.release()
        self.terminate()

    """convert cv format to qt"""
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return p
        #return QPixmap.fromImage(p)
    
    """draw RoI for hand"""
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
