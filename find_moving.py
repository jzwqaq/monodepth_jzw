# -*-coding:utf-8-*-
import cv2
import numpy as np
import copy


# def original(i, j, k, ksize, img):
#     # 找到矩阵坐标
#     x1 = y1 = -ksize // 2
#     x2 = y2 = ksize + x1
#     temp = np.zeros(ksize * ksize)
#     count = 0
#     # 处理图像
#     for m in range(x1, x2):
#         for n in range(y1, y2):
#             if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
#                 temp[count] = img[i, j, k]
#             else:
#                 temp[count] = img[i + m, j + n, k]
#             count += 1
#     return temp


# 自定义最大值滤波器最小值滤波器
# def max_min_functin(ksize, img, flag):
#     img0 = copy.copy(img)
#     for i in range(0, img.shape[0]):
#         for j in range(2, img.shape[1]):
#             for k in range(img.shape[2]):
#                 temp = original(i, j, k, ksize, img0)
#                 if flag == 0:   # 设置flag参数，如果是0就检测最大值，如果是1就检测最小值。
#                     img[i, j, k] = np.max(temp)
#                 elif flag == 1:
#                     img[i, j, k] = np.min(temp)
#     return img


source = cv2.imread("/home/robot/monodepth2/kitti_data/2011_09_28/2011_09_28_drive_0002_sync/image_02/data/"
                    "0000000015.png")
target = cv2.imread("/home/robot/monodepth2/kitti_data/2011_09_28/2011_09_28_drive_0002_sync/image_02/data/"
                    "0000000016.png")
prvs = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(source)
hsv[..., 1] = 255
next = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# flow2 = cv2.DenseOpticalFlow(prvs, next)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2  # 色调范围：0°~360°
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('frame1', rgb)
img = cv2.medianBlur(prvs - next, 5)
cv2.imshow('frame', img)
# cv2.imshow('frame2', flow2)

cv2.imshow("source", source)
cv2.imshow("target",  target)


cv2.waitKey(0)
