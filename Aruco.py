import csv
import ml_scale_use
import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
import pandas as pd



#加载鱼眼镜头的yaml标定文件
#加载相机纠正参数
global frame_pass,ret

def main_calibration():

    cv_file = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("dist_coeff").mat()
    cv_file.release()
    #加载默认cam参数
    dist=np.array(([[-0.01337232,  0.01314211, -0.00060755, -0.00497024,  0.08519319]]))
    newcameramtx=np.array([[484.55267334,   0.,         325.60812827],
    [  0.,         480.50973511, 258.93040826],
    [  0.,           0.,           1.        ]])
    mtx=np.array( [[428.03839374,   0,         339.37509535],
    [  0.,         427.81724311, 244.15085121],
    [  0.,           0.,          1.        ]])

    ##以上数据基于相机以及其他py程序


    cap = cv2.VideoCapture(1)
    #cap.set(3, 1280)  # 设置分辨率
    #cap.set(4, 768)


    #0:computer cam/1:usb cam
    font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
    CACHED_PTS = None
    CACHED_IDS = None
    Line_Pts = None
    frame_pass = None
    distance_x_total = 0
    distance_y_total = 0
    t = 1
    while cap.isOpened():
        Dist = []
        ret, frame = cap.read()
        if ret:
            assert not isinstance(frame, type(None)), 'frame not found'
        # 获取视频帧的高

        h1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取视频帧的宽
        w1 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(h1/2)
        w2 = int(w1 / 2)
        #print(h1,w1,h2,w2) #480 640 240 320
        cam_coordinate=(h2,w2)
        # 纠正畸变
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1),1)
        frame = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)

        cv2.line(frame, (w2 - 5, h2), (w2 + 5, h2), (0, 0, 225), 1)
        cv2.line(frame, (w2, h2 - 5), (w2, h2 + 5), (0, 0, 225), 1)

        #x, y, w1, h1 = roi
        #dst1 = dst1[y:y + h1, x:x + w1]

        #灰度化，检测aruco标签，所用字典为DICT_ARUCO_ORIGINAL
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, gray_otsu = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #retval 最合适阈值
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        parameters =  aruco.DetectorParameters_create()

        #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray_otsu,aruco_dict,parameters=parameters)
        #print(corners)
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
                #print("[INFO] Marker detected")
                corners_abcd = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                #print(corners_abcd)
                topRightPoint = (float(topRight[0]), float(topRight[1]))
                topLeftPoint = (float(topLeft[0]), float(topLeft[1]))
                bottomRightPoint = (float(bottomRight[0]), float(bottomRight[1]))
                bottomLeftPoint = (float(bottomLeft[0]), float(bottomLeft[1]))

                cX = ((topLeft[0] + bottomRight[0]) // 2)
                cY = ((topLeft[1] + bottomRight[1]) // 2)
                measure = abs(3.5/(topLeft[0]-cX))
                cv2.circle(frame, (int(cX), int(cY)), 4, (255, 0, 0), -1)
                #cv2.putText(frame, str(
                #    int(markerId)), (int(topLeft[0]-10), int(topLeft[1]-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                Dist.append((int(cX), int(cY)))
                # print(arucoDict)

                if len(Dist) == 0:
                    if Line_Pts is not None:
                       Dist = Line_Pts
                if len(Dist) == 2:
                    Line_Pts = Dist
                if len(Dist) == 2:
                    cv2.line(frame, Dist[0], Dist[1], (255, 0, 255), 2)
                    ed = (1/7.5)*1.8*((Dist[0][0] - Dist[1][0])**2 + ((Dist[0][1] - Dist[1][1])**2))**(0.5)
                    cv2.putText(frame, str(float(measure*(ed))) + "cm", (int(300), int(
                  300)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))


        if ids is not None:

            #获取aruco返回的rvec旋转矩阵、tvec位移矩阵
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
            # 估计每个标记的姿态并返回值rvet和tvec ---不同
            #rvec为旋转矩阵，tvec为位移矩阵
            # from camera coeficcients
            (rvec-tvec).any() # get rid of that nasty numpy value array error
            #print(rvec)


            #在画面上 标注auruco标签的各轴
            for i in range(rvec.shape[0]):
                aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners,ids)
                #aruco.drawAxis(gray_otsu, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 0.03)
                #aruco.drawDetectedMarkers(gray_otsu, corners, ids)
                c = corners[i][0]
                cx=float(c[:, 0].mean())
                cy=float(c[:, 1].mean())
                coordinate = (cx, cy)
                cv2.circle(frame, (int(cx),int(cy)), 2, (255, 255, 0), 2)
                print(cx,cy)
                #cv2.circle(gray_otsu, (int(cx), int(cy)), 2, (255, 255, 0), 2)

                #marker 中心与画面中心距离
                p1 = np.array(cam_coordinate)
                p2 = np.array(coordinate)
                p3 = p2 - p1
                #print(p1,p2,p3)
                #dis = math.hypot(p3[0], p3[1])*1.8/abs(float(p3[0]))
                #print(cam_coordinate,coordinate,p3,dis,'CM')
                dis_pix = ((p3[0])**2 + (p3[1])**2)**(0.5)
                #print('x=',p3[0],'y=',p3[1])

                ml_scale_use.pix_scale(p3[0],p3[1])
                distance_x = float(p3[0]) * ml_scale_use.x_scale
                distance_y = float(p3[1]) * ml_scale_use.y_scale

                rightdis = abs(topRight[0] - bottomRight[0])
                #print('rightdis=', rightdis, 'd=', rightdis * ml_scale_use.y_scale)

                dis_cm = ((distance_x) ** 2 + (distance_y) ** 2) ** (0.5)

                print('distance between marker and center ', dis_pix, 'pix')
                print('x=', distance_x, 'y=', distance_y)
                #print('x_scale=', ml_scale_use.x_scale, 'y_scale=', ml_scale_use.y_scale)
                print('distance between marker and center ', dis_cm, 'cm')
                distance_x_total = distance_x_total + distance_x
                distance_y_total = distance_y_total + distance_y
                t = t+1
                values = [ids, float(distance_x), float(distance_y)]
                with open('markers_distance.csv', 'a+', newline='') as fp:
                    # 获取对象
                    write = csv.writer(fp)
                    # write.writerow(headers)
                    write.writerow(values)





            ###### 显示id标记 #####
            #cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            #cv2.putText(gray_otsu, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ###### 角度估计 #####
            #print(rvec)
            #考虑Z轴（蓝色）的角度
            deg=rvec[0][0][2]/math.pi*180
            #deg=rvec[0][0][2]/math.pi*180*90/104
            # 旋转矩阵到欧拉角
            R=np.zeros((3,3),dtype=np.float64)
            if np.size(rvec) == 3:
                cv2.Rodrigues(rvec,R)
            sy=math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
            singular=sy< 1e-6
            if not singular:#偏航，俯仰，滚动
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            # 偏航，俯仰，滚动换成角度
            rx = x * 180.0 / 3.141592653589793
            ry = y * 180.0 / 3.141592653589793
            rz = z * 180.0 / 3.141592653589793

            cv2.putText(frame,'deg_z:'+str(ry)+str('deg'),(0, 140), font, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            #cv2.putText(gray_otsu, 'deg_z:' + str(ry) + str('deg'), (0, 140), font, 1, (0, 255, 0), 2,
            #            cv2.LINE_AA)
            #print("偏航，俯仰，滚动",rx,ry,rz)


            ###### 距离估计 #####
            #print(tvec[0][0][2])
            #distance = ((tvec[0][0][2] + 0.02) * 0.018) * 100  # 单位是米
            #distance = (tvec[0][0][2]) * 100  # 单位是米


            # 显示距离
            #cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 1, (0, 255, 0), 2,
            #            cv2.LINE_AA)
            #cv2.putText(gray_otsu, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 1, (0, 255, 0), 2,
            #            cv2.LINE_AA)

            ####真实坐标换算####（to do）
            # print('rvec:',rvec,'tvec:',tvec)
            # # new_tvec=np.array([[-0.01361995],[-0.01003278],[0.62165339]])
            # # 将相机坐标转换为真实坐标
            # r_matrix, d = cv2.Rodrigues(rvec)
            # r_matrix = -np.linalg.inv(r_matrix)  # 相机旋转矩阵
            # c_matrix = np.dot(r_matrix, tvec)  # 相机位置矩阵

        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            #cv2.putText(gray_otsu, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 显示结果画面

       #cv2.imshow("frame", new_frame)

        #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)


        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.resize(frame, (0, 0), fx=1, fy=1,
                                 interpolation=cv2.INTER_NEAREST)

        cv2.imshow("frame",frame)
        cv2.imshow("otsu", gray_otsu)

        key = cv2.waitKey(1)
        #print(time.process_time())

        #if float(time.process_time()) >= float(calibration_time):
        #    print(time.process_time())
        #    print('timeout break...')
        #    cap.release()
        #    cv2.destroyAllWindows()
        #    fp.close()
        #    break

        if key == 27:         # 按esc键退出
            print('esc break...')
            cap.release()
            cv2.destroyAllWindows()
            fp.close()
            break

        if key == ord(' '):   # 按空格键保存
    #        num = num + 1
    #        filename = "frames_%s.jpg" % num  # 保存一张图像
            filename = str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, frame)

if __name__ == '__main__':
    calibration_time= pd.read_csv('markers_time.csv', usecols=['Time']).values
    calibration_time = calibration_time *1.5
    #print(calibration_time)
    main_calibration()


