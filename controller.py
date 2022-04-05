##Copyright
##written by HUI, CHEUNG YUEN
##Student of HKUST
##FYP, FINAL YEAR PROJECT
import argparse
import sys, os, time, csv, cv2
from cv2 import aruco
import torch
from torch.backends import cudnn
import random
import ml_scale_use
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from UI import Ui_MainWindow
from yoloui import Ui_Yolowindow
from signupui import Ui_Dialog

import pandas as pd
from pyzbar import pyzbar
import datetime
from lib.share import shareInfo

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# pyuic5 -x UI.ui -o UI.py

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.timer_camera_2 = QTimer()  # 初始化定时器
        self.timer_camera_1 = QTimer()
        self.timer_camera_3 = QTimer()
        self.timer_camera_Q = QTimer()
        self.timer_camera_Y = QTimer()
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = shareInfo.CAM_NUM
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        #push button
        self.ui.pushButton.setText('RESET')
        self.clicked_counter = 0
        self.ui.pushButton.clicked.connect(self.buttonClicked_reset)
        self.ui.pushButton_2.setText('START')
        self.clicked_counter_2 = 0
        self.ui.pushButton_2.clicked.connect(self.slotCameraButton_2)  # monitoring mode
        self.ui.pushButton_3.setText('EXIT!')
        self.clicked_counter_3 = 0
        self.ui.pushButton_3.clicked.connect(self.buttonClicked_exit)
        self.ui.pushButton_4.setText('START')
        self.clicked_counter_4 = 0
        self.ui.pushButton_4.clicked.connect(self.buttonClicked_start)
        self.ui.pushButton_5.setText('ENTER')
        self.clicked_counter_5 = 0
        self.ui.pushButton_5.clicked.connect(self.buttonClicked_enter)
        self.ui.pushButton_6.setText('START')
        self.clicked_counter_6 = 0
        self.ui.pushButton_6.clicked.connect(self.buttonClicked_scale)  # scale mode
        self.ui.pushButton_7.setText('ENTER')
        self.clicked_counter_7 = 0
        self.ui.pushButton_7.clicked.connect(self.buttonClicked_offset)  # OFFSET
        self.ui.pushButton_8.setText('RESET')
        self.clicked_counter_8 = 0
        self.ui.pushButton_8.clicked.connect(self.buttonClicked_reset_s)
        self.ui.pushButton_9.setText('ENTER')
        self.clicked_counter_9 = 0
        self.ui.pushButton_9.clicked.connect(self.buttonClicked_QR)
        self.ui.pushButton_10.setText('START')
        self.clicked_counter_10 = 0
        self.ui.pushButton_10.clicked.connect(self.buttonClicked_yolo)  # YOLO5

        #radiobutton
        self.ui.radioButton.setText('Calibration Mode')
        self.ui.radioButton_2.setText('Monitor Mode')
        self.ui.radioButton.toggled.connect(self.onClicked_C)
        self.ui.radioButton_2.toggled.connect(self.onClicked_M)
        self.ui.radioButton_3.toggled.connect(self.onClicked_S)
        self.ui.radioButton_4.toggled.connect(self.onClicked_G)
        self.ui.radioButton_5.toggled.connect(self.onClicked_Q)
        self.ui.radioButton_6.toggled.connect(self.onClicked_Y)
        #timer
        self.timer_camera_2.timeout.connect(self.show_camera_2)
        self.timer_camera_1.timeout.connect(self.show_camera_1)
        self.timer_camera_3.timeout.connect(self.show_camera_3)
        self.timer_camera_Q.timeout.connect(self.show_camera_Q)
        self.timer_camera_Y.timeout.connect(self.show_camera_Y)
        #labal frame
        self.ui.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.label_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ui.label_7.setFrameShape(QFrame.Box)
        self.ui.label_7.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgb(255, 255, 240);')
        self.ui.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.label_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ui.label_2.setFrameShape(QFrame.Box)
        self.ui.label_2.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgb(255, 255, 240);')

        self.Time_Enter = 0
        self.buttonClick = False
        self.buttonClick_2 = False
        self.buttonClicked_o = False
        self.buttonClicked_Q = False
        self.grau_butten = False
        self.offsetx = 0
        self.offsety = 0
        self.markersize = 1.8
        self.scaletime = 60

    ###Funtion one !!!
    def onClicked_C(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            msg = "Please enter calibration time"
            self.ui.label_2.setText(msg)
            self.grau_butten = False
            self.buttonClick = True
            self.buttonClick_2 = False
            os.system("ml_scale_calculate.py")
    def onClicked_G(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            msg = "It will display the gray-otsu frame"
            self.ui.label_2.setText(msg)
            self.grau_butten = True
            time.sleep(1)
            msg = "Please enter calibration time"
            self.ui.label_2.setText(msg)
            self.buttonClick = True
            self.buttonClick_2 = False
    def buttonClicked_start(self):
        if self.buttonClick == True:
            self.Time_Enter = int(self.ui.lineEdit.text())
            # print(self.Time_Enter)
            ## Need to run at the same time
            global maxtime
            maxtime = self.Time_Enter
            # self.startThread()
            # self.show_camera_1()
            if self.timer_camera_1.isActive() == False:
                with open('markers_distance.csv', 'r+') as fp:
                    fp.truncate()
                    headers = ['ids', 'distance_x', 'distance_y']
                    write = csv.writer(fp)
                    write.writerow(headers)
                    fp.close()
                self.openCamera()
            msg = "calibrating... Please Wait"
            self.ui.label_2.setText(msg)
            self.ui.progressBar.setMaximum(maxtime + 10)
            for i in range(maxtime + 10):
                QApplication.processEvents()
                time.sleep(1)
                self.ui.progressBar.setValue(i + 1)

            self.printxydata()
            msg = "Calculation Finished"
            self.ui.label_2.setText(msg)
            self.ui.lineEdit.clear()

            self.ui.progressBar.setValue(0)

        else:
            msgs = 'Choose a camera MODE first'
            self.ui.label_2.setText(msgs)
    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            if self.buttonClick == True:
                self.timebeg_1 = time.process_time()
                self.timer_camera_1.start(10)
            else:
                self.timer_camera_2.start(10)

        if self.buttonClick_2 == True:
            self.ui.pushButton_2.setText('STOP')
            msg = "Please click STOP to stop the camera"
            self.ui.label_2.setText(msg)
            # QApplication.processEvents()

        # 关闭摄像头
    def show_camera_1(self):
        # load cam data
        cv_file = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("camera_matrix").mat()
        dist_matrix = cv_file.getNode("dist_coeff").mat()
        cv_file.release()
        dist = np.array(([[-0.01337232, 0.01314211, -0.00060755, -0.00497024, 0.08519319]]))
        newcameramtx = np.array([[484.55267334, 0., 325.60812827],
                                 [0., 480.50973511, 258.93040826],
                                 [0., 0., 1.]])
        mtx = np.array([[428.03839374, 0, 339.37509535],
                        [0., 427.81724311, 244.15085121],
                        [0., 0., 1.]])
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
        CACHED_PTS = None
        CACHED_IDS = None
        Line_Pts = None

        Dist = []
        ret, frame = self.cap.read()
        h1 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取视频帧的宽
        w1 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(h1 / 2)
        w2 = int(w1 / 2)
        # print(h1, w1, h2, w2)  # 480 640 240 320
        cam_coordinate = (int(h2 + self.offsetx), int(w2 + self.offsety))
        # 纠正畸变
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
        frame = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, gray_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # retval 最合适阈值
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_otsu, aruco_dict, parameters=parameters)
        if len(corners) <= 0:
            if CACHED_PTS is not None:
                corners = CACHED_PTS
        if len(corners) > 0:
            CACHED_PTS = corners
            if ids is not None:
                ids = ids.flatten()
                CACHED_IDS = ids
            else:
                if CACHED_IDS is not None:
                    ids = CACHED_IDS
            if len(corners) < 2:
                if len(CACHED_PTS) >= 2:
                    corners = CACHED_PTS
            for (markerCorner, markerId) in zip(corners, ids):
                # print("[INFO] Marker detected")
                corners_abcd = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                cX = ((topLeft[0] + bottomRight[0]) // 2)
                cY = ((topLeft[1] + bottomRight[1]) // 2)
                measure = abs(3.5 / (topLeft[0] - cX))
                cv2.circle(frame, (int(cX), int(cY)), 4, (255, 0, 0), -1)
                Dist.append((int(cX), int(cY)))
                if len(Dist) == 0:
                    if Line_Pts is not None:
                        Dist = Line_Pts
                if len(Dist) == 2:
                    Line_Pts = Dist
                if len(Dist) == 2:
                    cv2.line(frame, Dist[0], Dist[1], (255, 0, 255), 2)
                    ed = (1 / 7.5) * 1.8 * ((Dist[0][0] - Dist[1][0]) ** 2 + ((Dist[0][1] - Dist[1][1]) ** 2)) ** (0.5)
                    cv2.putText(frame, str(float(measure * (ed))) + "cm", (int(300), int(
                        300)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                if ids is not None:
                    # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    # print(rvec)

                    # 在画面上 标注auruco标签的各轴
                    for i in range(rvec.shape[0]):
                        aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
                        aruco.drawDetectedMarkers(frame, corners, ids)
                        c = corners[i][0]
                        cx = float(c[:, 0].mean())
                        cy = float(c[:, 1].mean())
                        coordinate = (cx, cy)
                        cv2.circle(frame, (int(cx), int(cy)), 2, (255, 255, 0), 2)
                        # marker 中心与画面中心距离
                        p1 = np.array(cam_coordinate)
                        p2 = np.array(coordinate)
                        p3 = p2 - p1

                        ml_scale_use.pix_scale(p3[0], p3[1])
                        distance_x = float(p3[0]) * (ml_scale_use.x_scale / 1.8) * float(self.markersize)
                        distance_y = float(p3[1]) * (ml_scale_use.y_scale / 1.8) * float(self.markersize)
                        values = [ids, float(distance_x), float(distance_y)]
                        with open('markers_distance.csv', 'a+', newline='') as fp:
                            write = csv.writer(fp)
                            write.writerow(values)

                else:
                    ##### DRAW "NO IDS" #####
                    cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    self.ui.textBrowser.setText("NO IDS")
                cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                self.ui.textBrowser.setText(str(ids))
        if self.grau_butten == True:
            show = cv2.resize(gray_otsu, (640, 480))
        else:
            show = cv2.resize(frame, (640, 480))

        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        cv2.line(show, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show, (320, 235), (320, 245), (0, 0, 225), 1)
        show = cv2.rotate(show, cv2.ROTATE_90_CLOCKWISE)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage))
        if float(time.process_time()-self.timebeg_1) >= float(maxtime):
            self.closeCamera_1()
    def closeCamera_1(self):
        self.cap.release()
        self.timer_camera_1.stop()
        self.ui.label_7.clear()
        self.ui.textBrowser.clear()
        self.buttonClick_2 = False

    ###Funtion TWO !!!
    def onClicked_M(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            msg = "Please click START to start the camera"
            self.ui.label_2.setText(msg)
            self.buttonClick_2 = True
            self.buttonClick = False
    def slotCameraButton_2(self):
        if self.buttonClick_2 == True:
            if self.timer_camera_2.isActive() == False:
                # 打开摄像头并显示图像信息
                self.openCamera()
                # self.startThread()
            else:
                self.closeCamera_2()
    def show_camera_2(self):
        flag, self.image_2 = self.cap.read()
        QApplication.processEvents()

        show_2 = cv2.resize(self.image_2, (640, 480))
        show_2 = cv2.cvtColor(show_2, cv2.COLOR_BGR2RGB)
        cv2.line(show_2, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show_2, (320, 235), (320, 245), (0, 0, 225), 1)
        show_2 = cv2.rotate(show_2, cv2.ROTATE_90_CLOCKWISE)
        showImage_2 = QImage(show_2.data, show_2.shape[1], show_2.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage_2))
    def closeCamera_2(self):
        self.cap.release()
        self.timer_camera_2.stop()
        # self.ui.label_7.clear()
        msg = "Please click START to start the camera"
        self.ui.label_2.setText(msg)
        self.ui.pushButton_2.setText('START')
        self.ui.label_7.clear()
        # self.ui.lcdNumber.clear()
    ###Other Function (X)
    def startThread(self):
        self.work.start()

    ###setup
    ###EXUT
    def buttonClicked_exit(self):
        self.clicked_counter_3 += 1
        if self.clicked_counter_3 >= 1:
            sys.exit()
    ###choose cam
    def buttonClicked_enter(self):
        self.CAM_NUM = int(self.ui.lineEdit_2.text())
        shareInfo.CAM_NUM = self.CAM_NUM
    ###reset printed result
    def buttonClicked_reset(self):
        self.ui.label_5.setText("NULL")
        self.ui.label_6.setText("NULL")
    ###print result
    def printxydata(self):
        distance_x = pd.read_csv('markers_distance.csv', usecols=['distance_x']).values
        distance_y = pd.read_csv('markers_distance.csv', usecols=['distance_y']).values
        print('here', distance_x.mean(), distance_y.mean())
        self.ui.label_5.setText(str(distance_x.mean()))
        self.ui.label_6.setText(str(distance_y.mean()))
        with open('markers_distance.csv', 'r+') as fp:
            fp.truncate()
            fp.close()

    ###yolo5(new win)
    def onClicked_Y(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            msg = "Recognition Mode Using YOLOv5"
            self.ui.label_2.setText(msg)
            self.closeCamera_Y()
            self.buttonClicked_Y = True
    def openCamera_Y(self):
        flag = self.cap.open(self.CAM_NUM,cv2.CAP_DSHOW)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            self.timer_camera_Y.start(10)
        #open cam to scan QR-code//     qr-num
    def buttonClicked_yolo(self):
        if self.buttonClicked_Y == True:
            print("jump to yolo")
            #shareInfo.createWin = win_Register()
            #shareInfo.createWin.show()
            os.system("yolowin.py")
            """
            if self.timer_camera_Y.isAccreateWintive() == False:
                # 打开摄像头并显示图像信息
                self.ui.pushButton_10.setText('STOP')
                self.openCamera_Y()
                 # self.startThread()
            else:
                self.closeCamera_Y()
            """
    def show_camera_Y(self):
        flag, self.image_y = self.cap.read()
        QApplication.processEvents()

        show_y = cv2.resize(self.image_y, (640, 480))
        show_y = cv2.cvtColor(show_y, cv2.COLOR_BGR2RGB)
        cv2.line(show_y, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show_y, (320, 235), (320, 245), (0, 0, 225), 1)
        show_y = cv2.rotate(show_y, cv2.ROTATE_90_CLOCKWISE)
        showImage_y = QImage(show_y.data, show_y.shape[1], show_y.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage_y))
    def closeCamera_Y(self):
        self.cap.release()
        self.timer_camera_1.stop()
        self.timer_camera_2.stop()
        self.timer_camera_3.stop()
        self.timer_camera_Q.stop()
        self.timer_camera_Y.stop()
        msg = "Please click START to start the camera"
        self.ui.label_2.setText(msg)
        self.ui.pushButton_10.setText('START')
        self.ui.label_7.clear()

    ###offset settings
    def onClicked_S(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            msg = "Please Enter the offset between cam and machine arm\n" \
                  "X <---------(0,0) \n" \
                  "              |\n" \
                  "              v\n" \
                  "(480,640)     Y"
            self.ui.label_2.setText(msg)
            self.buttonClicked_o = True
    def buttonClicked_offset(self):
        if self.buttonClicked_o == True:
            self.offsetx = float(self.ui.lineEdit_3.text())
            self.offsety = float(self.ui.lineEdit_4.text())
            self.markersize = float(self.ui.lineEdit_6.text())
            self.scaletime = float(self.ui.lineEdit_5.text())
            msg = "Entered"
            self.ui.label_2.setText(msg)
        else:
            msg = "Choose Scale Mode First!"
            self.ui.label_2.setText(msg)
    def buttonClicked_reset_s(self):
        if self.buttonClicked_o == True:
            self.ui.label_17.setText("")
            self.ui.label_18.setText("")
            self.ui.lineEdit_3.setText("0")
            self.ui.lineEdit_4.setText("0")
            self.offsety = 0
            self.offsetx = 0
            self.markersize = 1.8
            self.scaletime = 60
            self.ui.lineEdit_5.setText("60")
            self.ui.lineEdit_6.setText("1.8")
            msg = "Reset!"
            self.ui.label_2.setText(msg)
        else:
            msg = "Choose Scale Mode First!"
            self.ui.label_2.setText(msg)
    def buttonClicked_scale(self):
        if self.buttonClicked_o == True:
            #print(self.timer_camera_3.isActive())
            msg = "scale calculating... Please Wait"
            self.ui.label_2.setText(msg)
            if self.timer_camera_3.isActive() == False:
                # 打开摄像头并显示图像信息
                with open('markers_data2.csv', 'r+') as fp:
                    fp.truncate()
                    headers = ['index','x','y','rightdis','leftdis','topdis','bottomdis','average4','averagecol','averagerow']
                    write = csv.writer(fp)
                    write.writerow(headers)
                    fp.close()
                self.opencam = True
                self.openCamera_3()
            else:
                self.opencam = False

            self.ui.progressBar.setMaximum(self.scaletime*3)
            for i in range(self.scaletime*3):
                QApplication.processEvents()
                time.sleep(1)
                self.ui.progressBar.setValue(i + 1)
                if self.opencam == False:
                    break

            if self.opencam == False:
                self.closeCamera_3()
        else:
            msg = "Choose Scale Mode First!"
            self.ui.label_2.setText(msg)
    def show_camera_3(self):
        # load cam data

        cv_file = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("camera_matrix").mat()
        dist_matrix = cv_file.getNode("dist_coeff").mat()
        cv_file.release()
        dist = np.array(([[-0.01337232, 0.01314211, -0.00060755, -0.00497024, 0.08519319]]))
        newcameramtx = np.array([[484.55267334, 0., 325.60812827],
                                 [0., 480.50973511, 258.93040826],
                                 [0., 0., 1.]])
        mtx = np.array([[428.03839374, 0, 339.37509535],
                        [0., 427.81724311, 244.15085121],
                        [0., 0., 1.]])
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
        CACHED_PTS = None
        CACHED_IDS = None
        Line_Pts = None
        Dist = []
        index = 0
        ret, frame = self.cap.read()
        QApplication.processEvents()
        h1 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取视频帧的宽
        w1 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(h1 / 2)
        w2 = int(w1 / 2)
        # print(h1, w1, h2, w2)  # 480 640 240 320
        cam_coordinate = (h2 + self.offsetx, w2 + self.offsety)
        # 纠正畸变

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
        frame = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, gray_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # retval 最合适阈值
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_otsu, aruco_dict, parameters=parameters)

        if len(corners) <= 0:
            if CACHED_PTS is not None:
                corners = CACHED_PTS
        if len(corners) > 0:
            CACHED_PTS = corners
            if ids is not None:
                ids = ids.flatten()
                CACHED_IDS = ids
            else:
                if CACHED_IDS is not None:
                    ids = CACHED_IDS
            if len(corners) < 2:
                if len(CACHED_PTS) >= 2:
                    corners = CACHED_PTS

            for (markerCorner, markerId) in zip(corners, ids):
                # print("[INFO] Marker detected")
                corners_abcd = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                cX = ((topLeft[0] + bottomRight[0]) // 2)
                cY = ((topLeft[1] + bottomRight[1]) // 2)
                measure = abs(3.5 / (topLeft[0] - cX))
                cv2.circle(frame, (int(cX), int(cY)), 4, (255, 0, 0), -1)
                Dist.append((int(cX), int(cY)))
                if len(Dist) == 0:
                    if Line_Pts is not None:
                        Dist = Line_Pts
                if len(Dist) == 2:
                    Line_Pts = Dist
                if len(Dist) == 2:
                    cv2.line(frame, Dist[0], Dist[1], (255, 0, 255), 2)
                    ed = (1 / 7.5) * 1.8 * ((Dist[0][0] - Dist[1][0]) ** 2 + ((Dist[0][1] - Dist[1][1]) ** 2)) ** (0.5)
                    cv2.putText(frame, str(float(measure * (ed))) + "cm", (int(300), int(
                        300)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                if ids is not None:
                    # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    # print(rvec)

                    # 在画面上 标注auruco标签的各轴
                    for i in range(rvec.shape[0]):
                        aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
                        aruco.drawDetectedMarkers(frame, corners, ids)
                        c = corners[i][0]
                        cx = float(c[:, 0].mean())
                        cy = float(c[:, 1].mean())
                        coordinate = (cx, cy)
                        cv2.circle(frame, (int(cx), int(cy)), 2, (255, 255, 0), 2)
                        # marker 中心与画面中心距离
                        p1 = np.array(cam_coordinate)
                        p2 = np.array(coordinate)
                        p3 = p2 - p1

                        rightdis = abs(topRight[0] - bottomRight[0])
                        leftdis = abs(topLeft[0] - bottomLeft[0])
                        topdis = abs(topLeft[1] - topRight[1])
                        bottomdis = abs(bottomLeft[1] - bottomRight[1])
                        average4 = (rightdis + leftdis + topdis + bottomdis) / 4
                        averagecol = (rightdis + leftdis) / 2
                        averagerow = (topdis + bottomdis) / 2

                        # print(rightdis, leftdis, topdis, bottomdis)
                        # index,x,y,rightdis,leftdis,topdis,bottomdis,average4,averagecol,averagerow
                        values = [index, cx, cy, topdis, bottomdis, rightdis, leftdis, average4, averagecol, averagerow]

                        #with open('markers_data2.csv', 'a+', newline='') as fp:
                        with open('markers_data2.csv', 'a+', newline='') as fp:
                            # 获取对象
                            write = csv.writer(fp)
                            #print(cx,cy)
                            # write.writerow(headers)
                            write.writerow(values)
                            index = index + 1

                else:
                    ##### DRAW "NO IDS" #####
                    cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
                    self.ui.textBrowser.setText("NO IDS")
                cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                #self.ui.textBrowser.setText(str(ids),"X:",str(cx),"Y:",str(cy))

        # flag, frame = self.cap.read()
        QApplication.processEvents()

        show = cv2.resize(frame, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        cv2.line(show, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show, (320, 235), (320, 245), (0, 0, 225), 1)
        show = cv2.rotate(show, cv2.ROTATE_90_CLOCKWISE)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage))
        print(float(time.process_time()-self.timebeg),float(self.scaletime))
        if float(time.process_time()-self.timebeg) >= float(self.scaletime-10):
            self.closeCamera_3()
    def closeCamera_3(self):
        self.cap.release()
        self.timer_camera_3.stop()
        msg = "Finished"
        self.ui.label_2.setText(msg)
        self.ui.pushButton_6.setText('START')
        self.ui.textBrowser.clear()
        self.ui.progressBar.setValue(0)
        self.ui.label_7.clear()
        #print(self.timer_camera_3.isActive())

        QApplication.processEvents()
    def openCamera_3(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            self.timebeg = time.process_time()
            self.timer_camera_3.start(10)

        if self.buttonClicked_o == True:
            self.ui.pushButton_6.setText('STOP')
            msg = "Please click STOP to stop the camera"
            self.ui.label_2.setText(msg)
        # 关闭摄像头

    #QR-CODE
    def onClicked_Q(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            msg = "Scan QR - CODE to set up the parameters"
            self.ui.label_2.setText(msg)
            self.buttonClicked_Q = True
            if self.timer_camera_Q.isActive() == False:
                #print("here")
                self.openCamera_Q()
                # self.startThread()
            else:
                self.closeCamera_Q()
    def openCamera_Q(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            self.timer_camera_Q.start(10)
        #open cam to scan QR-code//     qr-num
    def show_camera_Q(self):
        flag, self.image_q = self.cap.read()
        QApplication.processEvents()


        barcodes = pyzbar.decode(self.image_q)
        self.barcodeData = ''
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(self.image_q, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.barcodeData = str(barcode.data.decode("utf-8"))
            #print(self.barcodeData)

        if self.barcodeData != '' :
            barcodeType = barcode.type
            self.barcodeData_list = self.barcodeData.split(",")
            if self.barcodeData_list[0] == "O":
                self.mode = 0
                msg = "Calibration Mode - Original image"
                self.ui.label_2.setText(msg)
                time = self.barcodeData_list[1]
                self.ui.lineEdit.setText(time)
            elif self.barcodeData_list[0] == "G":
                self.mode = 1
                msg = "Calibration Mode - Gray-Otsu image"
                self.ui.label_2.setText(msg)
                time = self.barcodeData_list[1]
                self.ui.lineEdit.setText(time)
            elif self.barcodeData_list[0] == "S":
                self.mode = 2
                msg = "Scale Setting Mode"
                self.ui.label_2.setText(msg)
                offsetx = self.barcodeData_list[1]
                self.ui.lineEdit_3.setText(offsetx)
                offsety = self.barcodeData_list[2]
                self.ui.lineEdit_4.setText(offsety)
                size = self.barcodeData_list[3]
                self.ui.lineEdit_3.setText(size)
                time = self.barcodeData_list[4]
                self.ui.lineEdit_5.setText(time)
            elif self.barcodeData_list[0] == "M":
                self.mode = 3
                msg = "Monitor Mode"
                self.ui.label_2.setText(msg)

            with open('qrcode.csv', 'a+') as fp:
                headers = [self.barcodeData, barcodeType, datetime.datetime.now()]
                write = csv.writer(fp)
                write.writerow(headers)
                fp.close()
            text = "{} ({})".format(self.barcodeData, barcodeType)
            cv2.putText(self.image_q, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.ui.textBrowser.setText(text)

        show_q = cv2.resize(self.image_q, (640, 480))
        show_q = cv2.cvtColor(show_q, cv2.COLOR_BGR2RGB)
        cv2.line(show_q, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show_q, (320, 235), (320, 245), (0, 0, 225), 1)
        show_q = cv2.rotate(show_q, cv2.ROTATE_90_CLOCKWISE)
        showImage_q = QImage(show_q.data, show_q.shape[1], show_q.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage_q))
    def closeCamera_Q(self):
        self.cap.release()
        self.timer_camera_Q.stop()
        msg = "Finished"
        self.ui.label_2.setText(msg)
        self.ui.label_7.clear()
    def buttonClicked_QR(self):
        if self.buttonClicked_Q == True:
            self.closeCamera_Q()
            if self.mode == 0: ##O
                self.grau_butten = False
                self.buttonClick = True
                self.buttonClick_2 = False
                os.system("ml_scale_calculate.py")
                self.buttonClicked_start()
            elif self.mode == 1: ##G
                self.grau_butten = True
                self.buttonClick = True
                self.buttonClick_2 = False
                os.system("ml_scale_calculate.py")
                self.buttonClicked_start()
            elif self.mode == 2: ##S
                self.buttonClicked_o = True
                self.buttonClicked_scale()
            elif self.mode == 3: ##M

                self.buttonClick_2 = True
                self.buttonClick = False
                self.slotCameraButton_2()

        else:
            msg = "Choose QR Mode First!"
            self.ui.label_2.setText(msg)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    #window = yolo_rec()
    window.show()
    sys.exit(app.exec_())