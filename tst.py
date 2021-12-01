# -*-coding:utf-8-*-
# import torch
# import numpy as np
# import cv2
# from layers import *

#
# # def tensor_to_np(tensor):
# #     img = tensor.mul(255).byte()
# #     # print(type(img), img.shape)
# #     img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
# #     # print(type(img), img.shape)
# #     return img
#
#
# a = torch.randint(0, 255, (1, 3, 192, 640))
# print(a.shape)
# im = tensor_to_np(a)
# cv2.imshow("img", im)
# cv2.waitKey(0)
# # b = a.squeeze(dim=0)
# # print(b.shape)
# # array = b.numpy()
# # print(array)
# # print(array.shape)
# # print(array.dtype)
# # img = np.transpose(array, (1, 2, 0)).astype(np.float32)

# -*-coding:utf-8-*-
import os
import sys
import cv2
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#######  解决CUDNN_STATUS_INTERNAL_ERROR问题  ######
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Root directory of the project
# ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco
import time

start = time.time()

# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "/home/robot/PycharmProjects/mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 192
    IMAGE_MAX_DIM = 640


config = InferenceConfig()
# config.display()
# Create model object in inference mode.
# with tf.device("/gpu:0"):
#     model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model = modellib.MaskRCNN(mode="inference", model_dir="../logs", config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

t1 = time.time()
print("load_time: ", t1 - start)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'person']

# Load a random image from the images folder
file_names = '/home/robot/monodepth2/kitti_data/2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000033.png'
# image = skimage.io.imread(file_names)
image = cv2.imread(file_names)
im = cv2.resize(image, [640, 192])
print(im.shape)
for i in range(50):
    t2 = time.time()
    # Run detection

    results = model.detect([im], verbose=1)
    t3 = time.time()
    print("detect_time:", t3 - t2)

# Visualize results
r = results[0]
print(r['masks'].shape)
masks = r['masks']
num_instances = masks.shape[2]  # 有多少个实例
# mask_array = np.moveaxis(masks, 0, -1)  # 移动shape的尺寸
mask_array_instance = []
output = np.zeros_like(im)
print(output.shape)
h, w, num = masks.shape
print("num:", num, "h:", h, "w:", w)
for i in range(num_instances):
    mask_array_instance.append(masks[:, :, i:(i+1)])
    print(mask_array_instance[i].shape)
    output = np.where(mask_array_instance[i]==True, 255, output)
mask = output
cv2.imshow('mask', mask)
cv2.waitKey(0)
visualize.display_instances(im, r['rois'], r['masks'], r['class_ids'], r['scores'])

