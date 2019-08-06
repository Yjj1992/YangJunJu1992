# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 08:59:03 2019
1. Recode all examples;

2. Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.
   Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.
   
@author: Ju
"""

import cv2
import random as rd
import numpy as np
#from matplotlib import pypolt as plt

#导入图片
img=cv2.imread('C:/Users/Ju/Downloads/lenna.jpg')

#图像随机截取函数
def img_crop(img):
    random_height1=rd.randint(0,img.shape[0]-400)
    random_height2=random_height1+400
    random_width1=rd.randint(0,img.shape[1]-400)
    random_width2=random_width1+400
    return img[random_height1:random_height2,random_width1:random_width2]

#图像随机换色
def img_color_shift(img):
    B,G,R=cv2.split(img)
    B, G, R = cv2.split(img)

    b_rand = rd.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = rd.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = rd.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    return img_merge

#图像旋转
def img_rotation(img):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1) # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate

#图像特征转化
def random_warp(img):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = rd.randint(-random_margin, random_margin)
    y1 = rd.randint(-random_margin, random_margin)
    x2 = rd.randint(width - random_margin - 1, width - 1)
    y2 = rd.randint(-random_margin, random_margin)
    x3 = rd.randint(width - random_margin - 1, width - 1)
    y3 = rd.randint(height - random_margin - 1, height - 1)
    x4 = rd.randint(-random_margin, random_margin)
    y4 = rd.randint(height - random_margin - 1, height - 1)

    dx1 = rd.randint(-random_margin, random_margin)
    dy1 = rd.randint(-random_margin, random_margin)
    dx2 = rd.randint(width - random_margin - 1, width - 1)
    dy2 = rd.randint(-random_margin, random_margin)
    dx3 = rd.randint(width - random_margin - 1, width - 1)
    dy3 = rd.randint(height - random_margin - 1, height - 1)
    dx4 = rd.randint(-random_margin, random_margin)
    dy4 = rd.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp

#图像增广
def augmentation_script(img):
    img_crop1=img_crop(img)
    img_color_shift1=img_color_shift(img_crop1)
    img_rotation1=img_rotation(img_color_shift1)
    img_random_warp1=random_warp(img_rotation1)
    return img_random_warp1

#测试
img_augmentation_script = augmentation_script(img)
cv2.imshow('lenna_augmentation_script', img_augmentation_script)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

        
    