# -*-coding:utf-8-*-
import numpy as np
import cv2
'''
代码：
第一步：使用cv2.capture读入视频
第二步：构造角点检测所需参数, 构造lucas kanade参数
第三步：拿到第一帧图像，并做灰度化， 作为光流检测的前一帧图像
第四步：使用cv2.goodFeaturesToTrack获得光流检测所需要的角点
第五步:构造一个mask用于画直线
第六步：读取一张图片，进行灰度化，作为光流检测的后一帧图像
第七步：使用cv2.caclOpticalFlowPyrLK进行光流检测
第八步：使用st==1获得运动后的角点，原始的角点位置
第九步：循环获得角点的位置，在mask图上画line，在后一帧图像上画角点
第十步：使用cv2.add()将mask和frame的像素点相加并进行展示
第十一步：使用后一帧的图像更新前一帧的图像，同时使用运动的后一帧的角点位置来代替光流检测需要的角点
'''
"""
calcOpticalFlowPyrLK.py:
由于目标对象或者摄像机的移动造成的图像对象在 续两帧图像中的移动 被称为光流。
它是一个 2D 向量场 可以用来显示一个点从第一帧图像到第二 帧图像之间的移动。
光 流在很多领域中都很有用
• 由运动重建结构
• 视频压缩
• Video Stabilization 等
"""

'''
重点函数解读：
1. cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  
用于获得光流估计所需要的角点
参数说明：
old_gray表示输入图片，
mask表示掩模，
feature_params:maxCorners=100角点的最大个数,
qualityLevel=0.3角点品质,minDistance=7即在这个范围内只存在一个品质最好的角点
2. pl, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  
用于获得光流检测后的角点位置
参数说明：pl表示光流检测后的角点位置，st表示是否是运动的角点，
err表示是否出错，old_gray表示输入前一帧图片，frame_gray表示后一帧图片，
p0表示需要检测的角点，lk_params：winSize表示选择多少个点进行u和v的求解，maxLevel表示空间金字塔的层数
3. cv2.add(frame, mask) # 将两个图像的像素进行加和操作
参数说明：frame表示输入图片，mask表示掩模
光流估计：通过当前时刻与前一时刻的亮度不变的特性
I(x, y, t) = I(x+?x, y+?y, t+?t) 使用lucas-kanade算法进行求解问题， 我们需要求得的是x,y方向的速度
'''



# 第一步：视频的读入
# cap = cv2.VideoCapture('../data/slow.flv')

# 第二步：构建角点检测所需参数
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
# maxLevel 为使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
source = cv2.imread("/home/robot/monodepth2/kitti_data/2011_09_28/2011_09_28_drive_0002_sync/image_02/data/"
                    "0000000063.jpg")
target = cv2.imread("/home/robot/monodepth2/kitti_data/2011_09_28/2011_09_28_drive_0002_sync/image_02/data/"
                    "0000000064.jpg")
# cv2.imshow("111", source)
# cv2.waitKey(0)
# 第三步：拿到第一帧图像并灰度化作为前一帧图片
# Take first frame and find corners in it
old_frame = source
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow("111", old_gray)
# cv2.waitKey(0)
# 第四步:返回所有检测特征点，需要输入图片，角点的最大数量，品质因子，minDistance=7如果这个角点里有比这个强的就不要这个弱的
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# 第五步:创建一个mask, 用于进行横线的绘制
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


# 第六步：读取图片灰度化作为后一张图片的输入
frame = target
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 第七步：进行光流检测需要输入前一帧和当前图像及前一帧检测到的角点
# calculate optical flow能够获取点的新位置
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
# 第八步：读取运动了的角点st == 1表示检测到的运动物体，即v和u表示为0
# Select good points
good_new = p1[st == 1]
good_old = p0[st == 1]
# 第九步：绘制轨迹
# draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 1)
    frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
# 第十步：将两个图片进行结合，并进行图片展示
img = cv2.add(frame, mask)
# cv2.resize
# img = cv2.resize(img, (1280, 640), interpolation=cv2.INTER_LINEAR)
cv2.namedWindow("frame", 0)
cv2.imshow('frame', img)

cv2.waitKey(0)  # & 0xff
# if k == 27:
#     break
# 第十一步：更新前一帧图片和角点的位置
# Now update the previous frame and previous points
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1, 1, 2)

# cv2.destroyAllWindows()
# cap.release()

