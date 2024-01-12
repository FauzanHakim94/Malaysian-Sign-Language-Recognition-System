import os
import sys
import cv2
import time
import atexit
from openpyxl import Workbook

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

import ViewTab
import RecordTab
import LiveTest

#alphabets = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
alphabets = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Bapa', 'Emak', 'Melayu', 'Cina', 'India']
rootPath = r'C:'
datasetPath = "dataset"

class UI(QMainWindow):
    def __init__(self):
        global alphabets, rootPath, datasetPath

        super(UI,self).__init__()
        ##Set root path
        rootPath = os.path.dirname(os.path.abspath(__file__))
        ##load the ui file
        loadUi("MSLRSGUI.ui", self)

        ###TOP Layout
        self.mainLogo = self.findChild(QLabel, "mainLogo")
        #show logo
        utemLogo = rootPath+"/supportGUI/UtemLogo.png"
        self.pixmap = QPixmap(utemLogo)
        self.mainLogo.setPixmap(self.pixmap)

        ###MIDDLE Layout
        self.searchLineEdit = self.findChild(QLineEdit, "filePathEditLine")
        self.searchButton = self.findChild(QPushButton, "searchButton")
        #widget connection
        self.searchButton.clicked.connect(self.searchButtonClicker)
        #display initial path
        datasetPath = rootPath+"\dataset"
        self.searchLineEdit.setText(datasetPath)

        ###BOTTOM Layout
        self.tabs = self.findChild(QTabWidget, "tabWidget")
        self.tabs.tabBarClicked.connect(self.tabClickedHandler)
        RecordTab.VTab.init(self, rootPath, alphabets, datasetPath)
        ViewTab.VTab.init(self, rootPath, datasetPath, alphabets)
        LiveTest.LTab.init(self, rootPath, datasetPath, alphabets)

        #show the app
        self.showMaximized()

    #file search function
    def searchFilePath(self, rootPath, fileName):
        for (root, dirs, files) in os.walk(rootPath, topdown=True):
            if root.find(fileName) != -1:
                break
        return root

    #when search button clicked
    def searchButtonClicker(self):
        global rootPath
        rootPath = str(QFileDialog.getExistingDirectory(self, 'Open File', rootPath))
        self.searchLineEdit.setText(rootPath)

    def tabClickedHandler(self, index):
        if index != 0:
            #close running thread on Record Tab
            RecordTab.cameraButtonState = 0
            RecordTab.VTab.threadState(self, RecordTab.cameraButtonState)
        if index != 1:
            pass
        if index != 2:
            pass
        if index != 3:
            #close running thread on Live Test
            LiveTest.cameraButtonState = 0
            LiveTest.LTab.threadState(self, LiveTest.cameraButtonState)

def OnExitApp(user):
    global datasetPath
    print(user, " exit Python application")
    if ViewTab.excelFlag:
        xb = ViewTab.wb
        xb.save(datasetPath + "\View Image Label.xlsx")
        print('manage to close excel')

#initialize the app
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
atexit.register(OnExitApp, user='FZ')
