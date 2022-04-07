# coding = utf-8
'''

date :  2022-4-1
desc:   实现眼部跟踪，控制鼠标移动位置

'''

import cv2
import dlib
import numpy as np
import os
import time
from imutils import face_utils
from tkinter import messagebox
import pyautogui as pag

from haversine import haversine

def createEyeMask(eyeLandmarks, im):
    # 创建眼睛蒙版
    leftEyePoints = eyeLandmarks
    eyeMask = np.zeros_like(im)
    cv2.fillConvexPoly(eyeMask, np.int32(leftEyePoints), (255, 255, 255))
    eyeMask = np.uint8(eyeMask)
    return eyeMask

def findIris(eyeMask, im, thresh):  #定位虹膜
    # 设定阈值来找到虹膜
    r = im[:,:,2]
    _, binaryIm = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    morph = cv2.dilate(binaryIm, kernel, 1)
    morph = cv2.merge((morph, morph, morph))
    morph = morph.astype(float)/255
    eyeMask = eyeMask.astype(float)/255
    iris = cv2.multiply(eyeMask, morph)
    #print("定位虹膜")
    #print(iris)
    return iris

def findCentroid(iris):  #计算质心
    # 寻找质心
    M = cv2.moments(iris[:,:,0])
    try:
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])

    except:
        cX = 0
        cY = 0

    centroid = (cX, cY)
    return centroid

def drawlines(img,IX,IY):  #画十字
    cv2.line(img, (IX - 3, IY), (IX + 3, IY), (0, 255, 255), 2)
    cv2.line(img, (IX , IY - 3), (IX, IY + 3), (0, 255, 255), 2)


pag.FAILSAFE = False
pwd = os.getcwd()   #获取当前文件夹路径
shape_detector_path = os.path.join(pwd,'shape_predictor_68_face_landmarks.dat')  #人脸特征点检测模型路径

faceDetector = dlib.get_frontal_face_detector()    #人脸检测器
landmarkDetector = dlib.shape_predictor(shape_detector_path)   #人脸特征点检测器

#对应特征点序号

RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END   = 42 - 1
LEFT_EYE_START  = 43 - 1
LEFT_EYE_END    = 48 - 1

cap = cv2.VideoCapture(0)
cnt = 0
while (1):
    ret,img = cap.read()  #读取视频流的一帧
    h,w = img.shape[0],img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
    rects = faceDetector(gray,0)   #人脸检测

    for rect in rects:   # 遍历每一个人脸
        #if rect > 1:
        #    messagebox.showinfo("窗口内不止一个人！")
        #    break
        shape = landmarkDetector(img, rect)  # 检测特征点
        points = face_utils.shape_to_np(shape)  # convert the facial(x,y)-coordinates to a Nump array
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
        leftrightEye = points[39:42 + 1]  # 取内眼角对应的特征点坐标
        eyecenterX = round((leftrightEye[0][0] + leftrightEye[3][0]) / 2)  # 计算内眼角X轴中点
        eyecenterY = round((leftrightEye[0][1] + leftrightEye[3][1]) / 2)  # 计算内眼角Y轴中点
        cv2.circle(img, (eyecenterX, eyecenterY), 4, (0, 255, 0), -1, 8)
        #需要解决头部位置改变后内眼角定位问题和改变量的计算，以及改变量与质心改变的关系

        leftEyeMask = createEyeMask(leftEye, img)
        rightEyeMask = createEyeMask(rightEye, img)
        # 设定阈值来找到虹膜
        leftIris = findIris(leftEyeMask, img, 40)
        rightIris = findIris(rightEyeMask, img, 50)

        # 寻找质心
        leftIrisCentroid = findCentroid(leftIris)
        rightIrisCentroid = findCentroid(rightIris)

        IrisCentroidX = round((leftIrisCentroid[0] + rightIrisCentroid[0]) / 2)  # 计算质心X轴中点
        IrisCentroidY = round((leftIrisCentroid[1] + rightIrisCentroid[1]) / 2)  # 计算质心Y轴中点

        drawlines(img, leftIrisCentroid[0], leftIrisCentroid[1])   #左眼瞳孔质心画十字
        drawlines(img, rightIrisCentroid[0], rightIrisCentroid[1])  #右眼瞳孔质心画十字
#        cv2.line(img, (rightIrisCentroid[0]-3, rightIrisCentroid[1]),(rightIrisCentroid[0]+3, rightIrisCentroid[1]), (0, 255, 255), 2);
#        cv2.line(img, (rightIrisCentroid[0], rightIrisCentroid[1]-3), (rightIrisCentroid[0], rightIrisCentroid[1]+3), (0, 255, 255), 2,);


        cv2.putText(img, "X0 = {0}".format(str(IrisCentroidX)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
        cv2.putText(img, "X1 = {0}".format(str(eyecenterX)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
        cv2.putText(img, "Y0 = {0}".format(str(IrisCentroidY)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
        cv2.putText(img, "Y1 = {0}".format(str(eyecenterY)), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
        cv2.putText(img, "WinWidth = {0}".format(img.shape[1]), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)
        cv2.putText(img, "WinHeight = {0}".format(img.shape[0]), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)

        cv2.circle(img, (IrisCentroidX, IrisCentroidY), 4, (0, 255, 0), -1, 8)
        #pag.moveTo(IrisCentroidX*3, IrisCentroidY*3)


    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cnt += 1
        cv2.imwrite("screenshoot" + str(cnt) + ".jpg", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


