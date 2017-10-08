import cv2
import numpy as np
import os
import math


def addGaussianNoise(image,variance):
    ims = image / 255.
    noise = np.float32(np.random.normal(0, variance, ims.shape))
    ims += noise
    ims = np.maximum(np.zeros(ims.shape), ims)
    ims = np.minimum(np.ones(ims.shape), ims)
    return np.array(ims * 255., dtype=np.uint8)

with open('/data4/szhou/224dataset/list_s2/face_train_r.txt') as o:
    pos_l = o.readlines()


#out = open('/data4/szhou/224dataset/list_s2/face_train_n.txt', 'w')




c = 0
for line in pos_l:
    c += 1
    path = os.path.join('/data4/szhou/', line.replace("\n", ""))
    #/data4/szhou/224dataset/Face/152480_00F40_Face.JPG 40
    pos = path.find(" ")
    score = path[pos:]
    path = path[:pos]
    #/data4/szhou/24dataset/Face/152480_00F40_Face.JPG
    src = cv2.imread(path)
    path = path.replace('/Face_r','/Temp')
    path = path.replace('/Face','/Temp')
    dst_path = path.replace('Temp','Face_n2')
    #/data4/szhou/224dataset/Face_n/152480_00F40_Face.JPG 
    #/data4/szhou/224dataset/Face_n/152480_00F40_Face_1.JPG 
    dst_path2 = dst_path
    #print dst_path
    #raw_path = dst_path2.split("/")[-1:][0]
    name1 = dst_path.replace(".JPG","_n1.JPG")
    name2 = dst_path.replace(".JPG","_n2.JPG")
    name3 = dst_path.replace(".JPG","_n3.JPG")
    name4 = dst_path.replace(".JPG","_n4.JPG")
    name5 = dst_path.replace(".JPG","_n5.JPG")
    #print name1
    #/data4/szhou/224dataset/Face_n/152480_00F40_Face_1_n1.JPG 
    # par_path = dst_path.replace("/"+raw_path,"")
    # #print dst_path
    # if not os.path.exists(par_path):
    #     os.makedirs(par_path)
    res1 = addGaussianNoise(src,0.001)
    res2 = addGaussianNoise(src,0.005)
    res3 = addGaussianNoise(src,0.01)
    res4 = addGaussianNoise(src,0.015)
    res5 = addGaussianNoise(src,0.02)
    cv2.imwrite(name1,res1)
    cv2.imwrite(name2,res2)
    cv2.imwrite(name3,res3)
    cv2.imwrite(name4,res4)
    cv2.imwrite(name5,res5)
    # out.write(name1.replace("/data4/szhou/","")+score+"\n")
    # out.write(name2.replace("/data4/szhou/","")+score+"\n")
    # out.write(name3.replace("/data4/szhou/","")+score+"\n")
    # out.write(name4.replace("/data4/szhou/","")+score+"\n")
    # out.write(name5.replace("/data4/szhou/","")+score+"\n")
    if c % 1000 == 0:
        print c
    # if c == 2:
    # 	break
print c