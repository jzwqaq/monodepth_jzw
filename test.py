# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
path = "/home/robot/下载/mono_1024x320_eigen.npy"
image = np.load(path)
print("image_size: ", image.shape[0])
for i in range(0, image.shape[0]):
    plt.cla()
    plt.imshow(image[i, :, :])
    plt.title('number: {}'.format(i), fontweight="bold")
    # plt.show()
    plt.pause(0.01)
https://github.com/nagadomi/distro.git
git clone https://github.com/torch/distro.git ~/torch --recursive
git clone https://github.com/nagadomi/distro.git ~/torch_nagadomi --recursive
