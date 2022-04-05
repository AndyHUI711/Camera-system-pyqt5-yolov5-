from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal


class LabelMouse(QLabel):
    double_clicked = pyqtSignal()


    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()

    def mouseMoveEvent(self):

        print('lable2')


class Label_click_Mouse(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()