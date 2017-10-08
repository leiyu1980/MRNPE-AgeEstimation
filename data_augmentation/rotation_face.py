import cv2
import numpy as np
import os
import math


def rotate_about_center(src, angle, scale=1.):
    rows,cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(src,M,(rows,cols))

with open('/data4/szhou/morph_224/temp_list/face.txt') as o:
    face_l = o.readlines()

c = 0

for line in face_l:
    c += 1
    path = os.path.join('/data4/szhou/morph_224/Face', line.replace('\n',''))
    src = cv2.imread(path)
    dst_path = path.replace('Face/','Face_r/')
    name1 = dst_path.replace(".JPG","_1.JPG")
    name2 = dst_path.replace(".JPG","_2.JPG")
    name3 = dst_path.replace(".JPG","_3.JPG")
    name4 = dst_path.replace(".JPG","_4.JPG")
    res1 = rotate_about_center(src,-8)
    res2 = rotate_about_center(src,-4)
    res3 = rotate_about_center(src,4)
    res4 = rotate_about_center(src,8)
    if not os.path.exists("/data4/szhou/morph_224/Face_r/"):
        os.makedirs("/data4/szhou/morph_224/Face_r/")
    cv2.imwrite(name1,res1)
    cv2.imwrite(name2,res2)
    cv2.imwrite(name3,res3)
    cv2.imwrite(name4,res4)
    if c % 1000 == 0:
        print c
print c
