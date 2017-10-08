import cv2
import os
import copy
# with open('/data4/szhou/224dataset/s2_test/s2_right_test.txt') as o:
#     pos_l = o.readlines()
# out = open('/data4/szhou/224dataset/s2_test/s2_right_test_m.txt', 'w')
with open('/data4/szhou/224dataset/list/Right_S1_Test.txt') as o:
    pos_l = o.readlines()
out = open('/data4/szhou/224dataset/list/Right_S1_Test_m.txt', 'w')
c = 0
for line in pos_l:
	c += 1
	path = os.path.join('/data4/szhou/', line.replace("\n", ""))
	pos = path.find(" ")
	score = path[pos:]
	path = path[:pos]
	img = cv2.imread(path)
	dst_path = path.replace('/RightEye','/RightEye_m')
	#image = cv2.LoadImage('lena.jpg',1)
	size = img.shape
	iLR = copy.deepcopy(img)
	h = size[0]  
	w = size[1]  
	for i in range(h): 
		for j in range(w):  
			iLR[i,w-1-j]=img[i,j]
	cv2.imwrite(dst_path,iLR)
	out.write(dst_path.replace("/data4/szhou/","")+score+"\n")
	if c%1000 == 0:
		print c
	#cv2.imwrite(new_name,iLR)  
	#cv2.imshow('image1',img) 
	#cv2.imshow('image2',iLR)  
	#cv2.waitKey(0)
print c