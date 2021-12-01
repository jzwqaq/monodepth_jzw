# coding :UTF-8
"""
Time:     2021/12/20 上午10:43
Author:   Jizhiwei
Version:  V 0.1
File:     3F2N_SNE.py
Describe: Writen during my master's degree at ZJUT
"""

import scipy.io as scio
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def visualization_map_creation(nx, ny, nz):
    [h, w] = nx.shape
    n_visual = np.zeros((h, w, 3))
    n_visual[:, :, 0] = nx
    n_visual[:, :, 1] = ny
    n_visual[:, :, 2] = nz
    n_visual = (1 - n_visual)/2
    return n_visual


def vector_normalization(nx, ny, nz):
    # temp = np.power(nx, 2) + np.power(ny, 2) + np.power(nz, 2)
    temp = nx**2 + ny**2 + nz**2
    mag = np.sqrt(temp)
    # print(mag)
    nxt = nx/mag
    nyt = ny/mag
    nzt = nz/mag
    return nxt, nyt, nzt


def delta_xyz_computation(X, Y, Z, pos):
    if pos == 0:
        kernel = [[0, -1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]
    elif pos == 1:
        kernel = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]
    elif pos == 2:
        kernel = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]
    elif pos == 3:
        kernel = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]
    elif pos == 4:
        kernel = [[-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]]
    elif pos == 5:
        kernel = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]
    elif pos == 6:
        kernel = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]
    else:
        kernel = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]

    X_d = signal.convolve2d(X, kernel, 'same')
    Y_d = signal.convolve2d(Y, kernel, 'same')
    Z_d = signal.convolve2d(Z, kernel, 'same')

    print("type(Z_d)", type(Z_d[100,100]))
    X_d[Z_d == 0] = np.nan
    Y_d[Z_d == 0] = np.nan
    Z_d[Z_d == 0] = np.nan
    return X_d, Y_d, Z_d
# todo    "load data "

# load depth map
path = "/home/robot/下载/mono_1024x320_eigen.npy"
image = np.load(path)
img = image[1, :, :]
print("type of data: ", type(img))
print("shape of data: ", img.shape)
Z = img
Z = Z.astype(np.float64)
[vmax, umax] = Z.shape
print(Z)
# cv2.imshow("depth", 255/Z)
plt.imshow(Z, cmap='gray')
plt.show()
cv2.waitKey(0)

# Z1 = data['Z']
# d1 = Z/256
# cv2.imshow("d1", d1)
# print("type(d1[200][200]): ", type(d1[200][200]))
# print("d1: ", d1[200][200])
# print("type(depth[200][200]): ", type(Z[200][200]))
# print("depth: ", Z[200][200])

# cv2.imshow("depth", Z)
# load surface_normal_ground_truth
surface_gt_File = "C:/Users/JZW\Desktop/Three-Filters-to-Normal-master/matlab_code/N.mat"
data = scio.loadmat(surface_gt_File)
N = data['N']
# print(pow(N[200][200][0], 2) + pow(N[200][200][1], 2) + pow(N[200][200][2], 2))
print("N.shape", N.shape)
nx_gt = N[:, :, 0]
print("nx_gt.shape", nx_gt.shape)
ny_gt = N[:, :, 1]
nz_gt = N[:, :, 2]
nx_gt, ny_gt, nz_gt = vector_normalization(nx_gt, ny_gt, nz_gt)
normal_gt = visualization_map_creation(-nx_gt, -ny_gt, -nz_gt)
# print("normal_gt: ", type(normal_gt[3, 3, 2]))
# print("sum: ", normal_gt[3, 3, 0]**2+normal_gt[3, 3, 1]**2+normal_gt[3, 3, 2]**2)
# print(normal_gt)
# cv2.cvtColor(normal_gt, normal_gt, )
# print(np.convolve([1, 2, 3], [0, 1, 0.5], 'same'))
# normal_gt = normal_gt.astype(np.uint8)
# print(img)
# cv2.imshow("normal_gt", normal_gt)
# # todo "初始化一些参数、变量"
third_filter = "median"  # mean median
kernel_size = 3  # 3*3 kernels
fx = 1400.0
fy = 1380.0
u0 = 350.0
v0 = 200.0
K = [[fx, 0, u0],
     [0, fy, v0],
     [0, 0, 1]]
print("K: ", K)
Gx = [[0, 0, 0],
      [-1, 0, 1],
      [0, 0, 0]]
Gy = [[0, -1, 0],
      [0, 0, 0],
      [0, 1, 0]]
print("Gx: ", Gx)
print("Gy: ", Gy)

# Z = Z.astype(np.float64)
# print(Z[200:250, 200:250])
# cv2.imshow("partofdepth", Z[:, 150:350])
X = np.zeros_like(Z)
Y = np.zeros_like(Z)

# 计算X,Y,Z

# print("type:", type(Z[200,200]), Z[200,200])
for u in range(Z.shape[0]):
    for v in range(Z.shape[1]):
        X[u][v] = Z[u][v] * (u+1 - u0) / fx
        Y[u][v] = Z[u][v] * (v+1 - v0) / fy
# print("X: **", X[100][100])
# todo：********** 3F2N ********
D = 1. / Z

# print(type(X[3, 3]))
Gu = signal.convolve2d(D, Gx, 'same')
Gv = signal.convolve2d(D, Gy, 'same')
# estimated nx and ny
nx_t = Gu * fx
ny_t = Gv * fy
# cv2.imshow("nxt", abs(nx_t))
# create a volume to compute nz
nz_t_volume = np.zeros((vmax, umax, 8))

for j in range(8):
    # print(j)
    [X_d, Y_d, Z_d] = delta_xyz_computation(X, Y, Z, j)
    # cv2.imshow("Z_d", Z_d*255)
    nz_j = -(nx_t * X_d + ny_t * Y_d)/Z_d
    nz_t_volume[:, :, j] = nz_j
    # cv2.imshow("nz", nz_j)
    # cv2.waitKey(0)
    # print(nz_j)

# print(len(Z_d[np.isnan(X_d)]))
nz_t = np.nanmedian(nz_t_volume, 2)
nx_t[np.isnan(nz_t)] = 0
ny_t[np.isnan(nz_t)] = 0
nz_t[np.isnan(nz_t)] = -1
# print(nz_t[200:220, 200:220])
[nx_t, ny_t, nz_t] = vector_normalization(nx_t, ny_t, nz_t)
# print(nx_t[210:212, 210:212])
# print(ny_t[210:212, 210:212])
# print(nz_t[210:212, 210:212])

nt_vis = visualization_map_creation(nx_t, ny_t, nz_t)
# print("x,y,z: ", nt_vis[50,50])
# nt_vis = nt_vis.astype(np.float32)
# img = cv2.cvtColor(nt_vis, code=cv2.COLOR_BGR2RGB)
# cv2.imshow("nt_vis", nt_vis)
# scio.savemat("nt_vis.mat", {'nt_vis': nt_vis})
plt.subplot(1, 2, 1)
plt.imshow(normal_gt)
plt.subplot(1, 2, 2)
plt.imshow(nt_vis)
c = plt.colorbar
plt.axis('off')
plt.show()
# cv2.waitKey(0)

