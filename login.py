##Copyright
##WrittenModified by HUI, CHEUNG YUEN
##Student of HKUST
##FYP, FINAL YEAR PROJECT
# -*- coding: utf-8 -*-
#reference1 https://github.com/ultralytics/yolov5
#reference2 https://github.com/BonesCat/YoloV5_PyQt5/tree/main
#reference3 https://github.com/Javacr/PyQt5-YOLOv5


from datetime import datetime

from utils.id_utils import get_id_info, sava_id_info

# login UI
from loginui import Ui_Login_Ui_Form
import csv
from cv2 import aruco
import ml_scale_use
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from UI import Ui_MainWindow
from signupui import Ui_Dialog

import pandas as pd
from pyzbar import pyzbar
import datetime
from lib.share import shareInfo

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox

from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window

# Login interface
class win_Login(QMainWindow):
    def __init__(self, parent = None):
        super(win_Login, self).__init__(parent)
        self.ui_login = Ui_Login_Ui_Form()
        self.ui_login.setupUi(self)
        self.init_slots()
        self.hidden_pwd()
    # hide password
    def hidden_pwd(self):
        self.ui_login.lineEdit_2.setEchoMode(QLineEdit.Password)
    # init
    def init_slots(self):
        self.ui_login.pushButton.clicked.connect(self.onSignIn)
        self.ui_login.lineEdit_2.returnPressed.connect(self.onSignIn) # enter
        self.ui_login.pushButton_2.clicked.connect(self.create_id)
    # Jump to sign up interface
    def create_id(self):
        print("jump to signup")
        shareInfo.createWin = win_Register()
        shareInfo.createWin.show()
    # Save login log
    def sava_login_log(self, username):
        with open('login_log.txt', 'a', encoding='utf-8') as f:
            f.write(username + '\t log in at' + datetime.now().strftimestrftime+ '\r')
    # login
    def onSignIn(self):
        print("You pressed sign in")
        # get the input
        username = self.ui_login.lineEdit.text().strip()
        password = self.ui_login.lineEdit_2.text().strip()
        # account info
        USER_PWD = get_id_info()
        # print(USER_PWD)
        if username not in USER_PWD.keys():
            replay = QMessageBox.warning(self,"!", "Incorrect username or password", QMessageBox.Yes)
        else:
            # if success
            if USER_PWD.get(username) == password:
                print("Jump to main window")
                shareInfo.loginWin = MainWindow_controller()
                shareInfo.loginWin.show()
                self.close()
            else:
                replay = QMessageBox.warning(self, "!", "Incorrect username or password", QMessageBox.Yes)
# signup interface
class win_Register(QMainWindow):
    def __init__(self, parent = None):
        super(win_Register, self).__init__(parent)
        self.ui_register = Ui_Dialog()
        self.ui_register.setupUi(self)
        self.init_slots()

    # init
    def init_slots(self):
        self.ui_register.pushButton_2.clicked.connect(self.new_account)
        self.ui_register.pushButton_3.clicked.connect(self.cancel)

    # new_account
    def new_account(self):
        print("Create new account")
        USER_PWD = get_id_info()
        # print(USER_PWD)
        new_username = self.ui_register.lineEdit.text().strip()
        new_password = self.ui_register.lineEdit_2.text().strip()
        new_password_2 = self.ui_register.lineEdit_3.text().strip()
        invicode = self.ui_register.lineEdit_4.text().strip()
        # Account already empty
        if new_username == "":
            replay = QMessageBox.warning(self, "!", "Username cannot be empty", QMessageBox.Yes)
        else:
            # Account already exists
            if new_username in USER_PWD.keys():
                replay = QMessageBox.warning(self, "!","Username already exists", QMessageBox.Yes)
            else:
                # pass
                if new_password == "":
                    replay = QMessageBox.warning(self, "!", "Password cannot be empty", QMessageBox.Yes)
                else:
                    if new_password_2 == "" or new_password_2 != new_password:
                        replay = QMessageBox.warning(self, "!", "Please confirm your password", QMessageBox.Yes)
                    else:
                        if invicode != "fyp2021wa03":# Successful
                            replay = QMessageBox.warning(self, "!", "Please confirm your Invitation Code", QMessageBox.Yes)
                        else:
                            print("Successful!")
                            sava_id_info(new_username, new_password)
                            replay = QMessageBox.warning(self, "!", "Successful！", QMessageBox.Yes)
                            # close
                            self.close()

    # cancellation
    def cancel(self):
        self.close()
# main functions// Calibration
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
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
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
            shareInfo.loginWin = MainWindow()
            shareInfo.loginWin.show()
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
#yolo thread
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit signals
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5n.pt'           # weight
        self.current_weight = './yolov5n.pt'    # weight
        self.source = '1'                       # cam
        self.conf_thres = 0.25                  # conf
        self.iou_thres = 0.45                   # iou
        self.jump_out = False                   # stop
        self.is_continue = True                 # pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # dalay OPEN/NO
        self.rate = 100                         # dalay HZ

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            print("test")
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0

            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('STOP!')
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                # stop
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1

                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness)


                    if self.rate_check:
                        time.sleep(1/self.rate)
                    # print(type(im0s))
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('Finish')

                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)
#yolo interface
class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)


        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # model setup
        self.ComboBox_weight.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.ComboBox_weight.clear()
        self.ComboBox_weight.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5
        self.det_thread = DetThread()
        self.model_type = self.ComboBox_weight.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '1'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)
        self.pushButton.clicked.connect(self.reset)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)
        self.confSlider.setValue(25)
        self.confSpinBox.setValue(0.25)
        self.iouSlider.setValue(45)
        self.iouSpinBox.setValue(0.45)

        self.ComboBox_weight.currentTextChanged.connect(self.change_model)
        # self.ComboBox_weight.currentTextChanged.connect(lambda x: self.statistic_msg('model %s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()
    def reset(self):
        self.search_pt()
        self.stop()
        iou = 0.25
        conf = 0.45
        rate = 10
        check = 0
        self.det_thread.source = '1'
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        MessageBox(
            self.closeButton, title='WARNING！', text='ALL PARAMETERS WILL BE RESET', time=2000, auto=True).exec_()
        self.statistic_msg('RESET!!!')
    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ComboBox_weight.clear()
            self.ComboBox_weight.addItems(self.pt_list)

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='！', text='Please wait', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='！', text='Please wait', time=2000, auto=True).exec_()
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading Cam：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)


    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.25
            conf = 0.45
            rate = 10
            check = 0
            new_config = {"iou": 0.25,
                          "conf": 0.45,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            iou = config['iou']
            conf = config['conf']
            rate = config['rate']
            check = config['check']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
            print(x)
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
            print(x)
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
            print(x)
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
            print(x)
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
            print(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
            print(x)
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def change_model(self, x):
        self.model_type = self.ComboBox_weight.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Model change to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Choose Pic or Video', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loading file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()


    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'CAM' if source.isnumeric() else source
            self.statistic_msg('Detecting >> Model：{}，Input File：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('PAUSE')

    def stop(self):
        self.det_thread.jump_out = True
        self.out_video.clear()
        self.raw_video.clear()

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))


    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            if results != []:
                self.resultWidget.setStyleSheet("background-color:red")
                self.resultWidget.addItems(results)
            else:
                self.resultWidget.setStyleSheet("background-color: rgba(12, 28, 77, 0)")
                self.resultWidget.addItems(results)


        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):

        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        shareInfo.loginWin = MainWindow_controller()
        shareInfo.loginWin.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    shareInfo.loginWin = win_Login()
    shareInfo.loginWin.show()
    sys.exit(app.exec_())

