import ui.RecordTab as RecordTab
import ui.ViewTab as ViewTab
import ui.LiveTest as LiveTest

import os
import sys
import atexit

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

TITLE = "Malaysian Sign Language Recognition"
SUBTITLE = "SUPERVISOR: Assoc. Prof. Dr. Masrullizam Bin Mat Ibrahim"
SUBTITLE2 = "DEVELEPOR: Muhammad Fauzan Bin Abdul Hakim"
ALPHABETS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Bapa', 'Emak', 'Melayu', 'Cina', 'India']
DATASET_PATH = "/dataset"
GUI_PATH = "ui/MSLRSGUI.ui"
MAIN_LOGO = "/ui_assets/UtemLogo.png"

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        loadUi(GUI_PATH, self)

        self.rootPath = os.path.dirname(os.path.abspath(__file__))
        mainLogoPath = self.rootPath+MAIN_LOGO
        datasetPath = self.rootPath+DATASET_PATH      

        # Widget List
        self.mainLogo = self.findChild(QLabel, "mainLogo")
        self.title = self.findChild(QLabel, "Title")        
        self.subtitle = self.findChild(QLabel, "subTitle")
        self.subtitle2 = self.findChild(QLabel, "subTitle2")                
        self.searchLineEdit = self.findChild(QLineEdit, "filePathEditLine")
        self.searchButton = self.findChild(QPushButton, "searchButton")
        self.tabs = self.findChild(QTabWidget, "tabWidget")

        # Widget Connection
        self.searchButton.clicked.connect(self.searchButtonClicker)
        self.tabs.tabBarClicked.connect(self.tabClickedHandler)

        # Initial Display
        self.pixmap = QPixmap(mainLogoPath)
        self.mainLogo.setPixmap(self.pixmap)
        self.searchLineEdit.setText(datasetPath)
        self.title.setText(TITLE)
        self.subtitle.setText(SUBTITLE)
        self.subtitle2.setText(SUBTITLE2)

        # Setup All Tabs
        RecordTab.RTab.init(self, self.rootPath, ALPHABETS, datasetPath)
        # ViewTab.VTab.init(self, ROOT_PATH, DATASET_PATH, ALPHABETS)
        # LiveTest.LTab.init(self, ROOT_PATH, DATASET_PATH, ALPHABETS)

        self.showMaximized()

    """when search button clicked"""
    def searchButtonClicker(self):
        self.rootPath = str(QFileDialog.getExistingDirectory(self, 'Open File', self.rootPath))
        self.searchLineEdit.setText(self.rootPath)

    def tabClickedHandler(self, index):
        if index != 0:
            # Close running thread on Record Tab
            RecordTab.cameraButtonState = 0
            RecordTab.RTab.threadState(self, RecordTab.cameraButtonState)
        if index != 1:
            pass
        if index != 2:
            pass
        if index != 3:
            #close running thread on Live Test
            # LiveTest.cameraButtonState = 0
            # LiveTest.LTab.threadState(self, LiveTest.cameraButtonState)
            print(f'index!=3')

# Start the App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
