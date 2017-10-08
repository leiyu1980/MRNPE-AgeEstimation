import numpy as np
import sys
# caffe_root = 'F:/liuting/caffe-revise/'
# sys.path.insert(0, caffe_root + 'python')
import caffe  
import PIL
from PIL import Image
import cv2

IMAGE_TXT1 = '/data4/szhou/CACD/test/face_test_s.txt'
IMAGE_TXT2 = '/data4/szhou/CACD/test/left_test_s.txt'
IMAGE_TXT3 = '/data4/szhou/CACD/test/nose_test_s.txt'
IMAGE_TXT4 = '/data4/szhou/CACD/test/mouth_test_s.txt'
IMAGE_TXT1_m = '/data4/szhou/CACD/test/face_test_m.txt'
IMAGE_TXT2_m = '/data4/szhou/CACD/test/left_test_m.txt'
IMAGE_TXT3_m = '/data4/szhou/CACD/test/nose_test_m.txt'
IMAGE_TXT4_m = '/data4/szhou/CACD/test/mouth_test_m.txt'
PREDICT_RESULT = '/data4/szhou/CACD/test/CACD_result.txt'
parent_dir = '/data4/szhou/CACD/'
MODEL_FILE1 = '/data/szhou/CACD/eye_net/deploy1.prototxt'
MODEL_FILE2 = '/data/szhou/CACD/nose_net/deploy1.prototxt'
MODEL_FILE3 = '/data/szhou/CACD/mouth_net/deploy1.prototxt'
PRETRAINED1 = '/data/szhou/CACD/eye_net/snapshot/age_iter_68000.caffemodel'
PRETRAINED2 = '/data/szhou/CACD/nose_net/snapshot/age_iter_80000.caffemodel'
PRETRAINED3 = '/data/szhou/CACD/mouth_net/snapshot/age_iter_46000.caffemodel'

# IMAGE_TXT1 = '/data4/szhou/CACD/test/face_test_s.txt'
# IMAGE_TXT2 = '/data4/szhou/CACD/test/left_test_s.txt'
# IMAGE_TXT3 = '/data4/szhou/CACD/test/nose_test_s.txt'
# IMAGE_TXT4 = '/data4/szhou/CACD/test/mouth_test_s.txt'
# IMAGE_TXT1_m = '/data4/szhou/CACD/test/face_test_m.txt'
# IMAGE_TXT2_m = '/data4/szhou/CACD/test/left_test_m.txt'
# IMAGE_TXT3_m = '/data4/szhou/CACD/test/nose_test_m.txt'
# IMAGE_TXT4_m = '/data4/szhou/CACD/test/mouth_test_m.txt'
# PREDICT_RESULT = '/data4/szhou/CACD/test/CACD_result_vgg.txt'
# parent_dir = '/data4/szhou/CACD/'
# MODEL_FILE1 = '/data/szhou/CACD/vgg_eye_net/age_deploy.prototxt'
# MODEL_FILE2 = '/data/szhou/CACD/vgg_nose_net/age_deploy.prototxt'
# MODEL_FILE3 = '/data/szhou/CACD/vgg_mouth_net/age_deploy.prototxt'
# PRETRAINED1 = '/data/szhou/CACD/vgg_eye_net/snapshot/age_iter_47000.caffemodel'
# PRETRAINED2 = '/data/szhou/CACD/vgg_nose_net/snapshot/age_iter_49000.caffemodel' 
# PRETRAINED3 = '/data/szhou/CACD/vgg_mouth_net/snapshot/age_iter_45000.caffemodel'

# This is MRNPRE(VGG) for MORPH S2,S1+S3
# IMAGE_TXT1 = '/data4/szhou/224dataset/s2_test/s2_face_test.txt'
# IMAGE_TXT2 = '/data4/szhou/224dataset/s2_test/s2_left_test.txt'
# IMAGE_TXT3 = '/data4/szhou/224dataset/s2_test/s2_nose_test.txt'
# IMAGE_TXT4 = '/data4/szhou/224dataset/s2_test/s2_mouth_test.txt'
# IMAGE_TXT1_m = '/data4/szhou/224dataset/s2_test/s2_face_test_m.txt'
# IMAGE_TXT2_m = '/data4/szhou/224dataset/s2_test/s2_left_test_m.txt'
# IMAGE_TXT3_m = '/data4/szhou/224dataset/s2_test/s2_nose_test_m.txt'
# IMAGE_TXT4_m = '/data4/szhou/224dataset/s2_test/s2_mouth_test_m.txt'
# # PREDICT_RESULT = '/data4/szhou/224dataset/result/s2/vgg_all_1.txt'
# PREDICT_RESULT = '/fastdata/caffe-rc4/python/result_vgg_s2'
# parent_dir = '/data4/szhou/'
# MODEL_FILE1 = '/data/szhou/224dataset/vgg_net_s2/eye_net/age_deploy.prototxt'
# MODEL_FILE2 = '/data/szhou/224dataset/vgg_net_s2/nose_net/age_deploy.prototxt'
# MODEL_FILE3 = '/data/szhou/224dataset/vgg_net_s2/mouth_net/age_deploy.prototxt'
# PRETRAINED1 = '/data/szhou/224dataset/vgg_net_s2/eye_net/snapshot/age_iter_15000.caffemodel'
# PRETRAINED2 = '/data/szhou/224dataset/vgg_net_s2/nose_net/snapshot/age_iter_13000.caffemodel'
# PRETRAINED3 = '/data/szhou/224dataset/vgg_net_s2/mouth_net/snapshot/age_iter_14000.caffemodel'


# This is MRNPRE(VGG) for MORPH S1,S2+S3
# IMAGE_TXT1 = '/data4/szhou/224dataset/list/Face_S1_Test.txt'
# IMAGE_TXT2 = '/data4/szhou/224dataset/list/Left_S1_Test.txt'
# IMAGE_TXT3 = '/data4/szhou/224dataset/list/Nose_S1_Test.txt'
# IMAGE_TXT4 = '/data4/szhou/224dataset/list/Mouth_S1_Test.txt'
# IMAGE_TXT1_m = '/data4/szhou/224dataset/list/Face_S1_Test_m.txt'
# IMAGE_TXT2_m = '/data4/szhou/224dataset/list/Left_S1_Test_m.txt'
# IMAGE_TXT3_m = '/data4/szhou/224dataset/list/Nose_S1_Test_m.txt'
# IMAGE_TXT4_m = '/data4/szhou/224dataset/list/Mouth_S1_Test_m.txt'
# # PREDICT_RESULT = '/data4/szhou/224dataset/result/s1/vgg_all_2.txt'
# PREDICT_RESULT = '/fastdata/caffe-rc4/python/result_vgg_s1'
# parent_dir = '/data4/szhou/'
# MODEL_FILE1 = '/data/szhou/224dataset/vgg_net/eye_net/age_deploy.prototxt'
# MODEL_FILE2 = '/data/szhou/224dataset/vgg_net/nose_net/age_deploy.prototxt'
# MODEL_FILE3 = '/data/szhou/224dataset/vgg_net/mouth_net/age_deploy.prototxt'
# PRETRAINED1 = '/data/szhou/224dataset/vgg_net/eye_net/snapshot/age_iter_13000.caffemodel'
# PRETRAINED2 = '/data/szhou/224dataset/vgg_net/nose_net/snapshot/age_iter_13000.caffemodel'
# PRETRAINED3 = '/data/szhou/224dataset/vgg_net/mouth_net/snapshot/age_iter_13000.caffemodel'

# This is MRNPRE(Alexnet) for MORPH S1,S2+S3
IMAGE_TXT1 = '/data4/szhou/224dataset/list/Face_S1_Test.txt'
IMAGE_TXT2 = '/data4/szhou/224dataset/list/Left_S1_Test.txt'
IMAGE_TXT3 = '/data4/szhou/224dataset/list/Nose_S1_Test.txt'
IMAGE_TXT4 = '/data4/szhou/224dataset/list/Mouth_S1_Test.txt'
IMAGE_TXT1_m = '/data4/szhou/224dataset/list/Face_S1_Test_m.txt'
IMAGE_TXT2_m = '/data4/szhou/224dataset/list/Left_S1_Test_m.txt'
IMAGE_TXT3_m = '/data4/szhou/224dataset/list/Nose_S1_Test_m.txt'
IMAGE_TXT4_m = '/data4/szhou/224dataset/list/Mouth_S1_Test_m.txt'
PREDICT_RESULT = '/data4/szhou/224dataset/result/s1/result_left_mouth.txt'
parent_dir = '/data4/szhou/'
MODEL_FILE1 = '/data/szhou/224dataset/net/eye_net/deploy1.prototxt'
MODEL_FILE2 = '/data/szhou/224dataset/net/nose_net/deploy1.prototxt'
MODEL_FILE3 = '/data/szhou/224dataset/net/mouth_net/deploy1.prototxt'
PRETRAINED1 = '/data/szhou/224dataset/net/eye_net/snapshot/age_iter_34000.caffemodel'
PRETRAINED2 = '/data/szhou/224dataset/net/nose_net/snapshot/age_iter_23000.caffemodel'
PRETRAINED3 = '/data/szhou/224dataset/net/mouth_net/snapshot/age_iter_23000.caffemodel'

# This is MRNPRE(Alexnet) for MORPH S2,S1+S3
# IMAGE_TXT1 = '/data4/szhou/224dataset/s2_test/s2_face_test.txt'
# IMAGE_TXT2 = '/data4/szhou/224dataset/s2_test/s2_left_test.txt'
# IMAGE_TXT3 = '/data4/szhou/224dataset/s2_test/s2_nose_test.txt'
# IMAGE_TXT4 = '/data4/szhou/224dataset/s2_test/s2_mouth_test.txt'
# IMAGE_TXT1_m = '/data4/szhou/224dataset/s2_test/s2_face_test_m.txt'
# IMAGE_TXT2_m = '/data4/szhou/224dataset/s2_test/s2_left_test_m.txt'
# IMAGE_TXT3_m = '/data4/szhou/224dataset/s2_test/s2_nose_test_m.txt'
# IMAGE_TXT4_m = '/data4/szhou/224dataset/s2_test/s2_mouth_test_m.txt'
# PREDICT_RESULT = '/data4/szhou/224dataset/result/s2/result2_left_mouth.txt'
# parent_dir = '/data4/szhou/'
# MODEL_FILE1 = '/data/szhou/224dataset/net_s2/eye_net/deploy1.prototxt'
# MODEL_FILE2 = '/data/szhou/224dataset/net_s2/nose_net/deploy1.prototxt'
# MODEL_FILE3 = '/data/szhou/224dataset/net_s2/mouth_net/deploy1.prototxt'
# PRETRAINED1 = '/data/szhou/224dataset/net_s2/eye_net/snapshot/age_iter_65000.caffemodel'
# PRETRAINED2 = '/data/szhou/224dataset/net_s2/nose_net/snapshot/age_iter_58000.caffemodel'
# PRETRAINED3 = '/data/szhou/224dataset/net_s2/mouth_net/snapshot/age_iter_58000.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)

net1 = caffe.Net(MODEL_FILE1, PRETRAINED1, caffe.TEST)
net2 = caffe.Net(MODEL_FILE2, PRETRAINED2, caffe.TEST)
net3 = caffe.Net(MODEL_FILE3, PRETRAINED3, caffe.TEST)
net1_m = caffe.Net(MODEL_FILE1, PRETRAINED1, caffe.TEST)
net2_m = caffe.Net(MODEL_FILE2, PRETRAINED2, caffe.TEST)
net3_m = caffe.Net(MODEL_FILE3, PRETRAINED3, caffe.TEST)


mae0_e = 0
mae0_n = 0
mae0_m = 0
num0 = 0
num11 = 0
num21 = 0
num31 = 0
num41 = 0
num51 = 0
num61 = 0
num71 = 0
num81 = 0
num91 = 0
mae11_e = 0
mae21_e = 0
mae31_e = 0
mae41_e = 0
mae51_e = 0
mae61_e = 0
mae71_e = 0
mae81_e = 0
mae91_e = 0
mae11_n = 0
mae21_n = 0
mae31_n = 0
mae41_n = 0
mae51_n = 0
mae61_n = 0
mae71_n = 0
mae81_n = 0
mae91_n = 0
mae11_m = 0
mae21_m = 0
mae31_m = 0
mae41_m = 0
mae51_m = 0
mae61_m = 0
mae71_m = 0
mae81_m = 0
mae91_m = 0
image_width = 224
image_height = 224


f1 = open(IMAGE_TXT1,'r')
f2 = open(IMAGE_TXT2,'r')
f3 = open(IMAGE_TXT3,'r')
f4 = open(IMAGE_TXT4,'r')
f5 = open(IMAGE_TXT1_m,'r')
f6 = open(IMAGE_TXT2_m,'r')
f7 = open(IMAGE_TXT3_m,'r')
f8 = open(IMAGE_TXT4_m,'r')
fsave = open(PREDICT_RESULT,'w')



img11 = np.zeros((1,3,image_height,image_width),np.float32)
img22 = np.zeros((1,3,image_height,image_width),np.float32)
img33 = np.zeros((1,3,image_height,image_width),np.float32)
img44 = np.zeros((1,3,image_height,image_width),np.float32)
img55 = np.zeros((1,3,image_height,image_width),np.float32)
img66 = np.zeros((1,3,image_height,image_width),np.float32)
img77 = np.zeros((1,3,image_height,image_width),np.float32)
img88 = np.zeros((1,3,image_height,image_width),np.float32)



# read lines form a file
c = 0
count = 0
diff_sum = 0
diff_sum1 = 0
diff_sum2 = 0
diff_sum3 = 0
diff_sum1_m = 0
diff_sum2_m = 0
diff_sum3_m = 0

for (line1,line2,line3,line4,line5,line6,line7,line8) in zip(f1,f2,f3,f4,f5,f6,f7,f8):  
   c += 1
   count = count +1
   idx1 = line1.find(' ')
   true_age = int(line1[idx1+1:len(line1)-1])
   #print ('true_age = %f' % float(true_age))
   fullName1 = parent_dir+line1[0:idx1]
   # print true_age
   # print fullName1
   #print fullName1
   img1 = cv2.imread(fullName1) #open image
   #img1 = cv2.resize(img1, (60,  60), interpolation=cv2.INTER_AREA) #resize test image to 256
   image = np.array(img1, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] #opencv read numpy h*w*c , caffe read n*c*h*w ( n =batch size ) channel[0] = blue
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img11[0,0,:,:] = b #batch size = 1 , so it is equal to zero, caffe is bgr too
   img11[0,1,:,:] = g
   img11[0,2,:,:] = r


   idx2 = line2.find(' ')
   true_age = int(line2[idx2+1:len(line2)-1])
   #print ('true_age = %f' % float(true_age))
   fullName2 = parent_dir+line2[0:idx2]
   # print true_age
   # print fullName1
   #print fullName1
   img2 = cv2.imread(fullName2) #open image
   #img2 = cv2.resize(img2, (60, 60), interpolation=cv2.INTER_AREA) #resize test image to 256
   image = np.array(img2, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] #opencv read numpy h*w*c , caffe read n*c*h*w ( n =batch size ) channel[0] = blue ,this is cv
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img22[0,0,:,:] = b #batch size = 1 , so it is equal to zero, caffe is bgr too , this is caffe
   img22[0,1,:,:] = g
   img22[0,2,:,:] = r
   
   idx3 = line3.find(' ')
   true_age = int(line3[idx3+1:len(line3)-1])
   #print ('true_age = %f' % float(true_age))
   fullName3 = parent_dir+line3[0:idx3]
   # print true_age
   # print fullName1
   #print fullName1
   img3 = cv2.imread(fullName3) #open image
   #img3 = cv2.resize(img3, (60, 60), interpolation=cv2.INTER_AREA) #resize test image to 256
   image = np.array(img3, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] #opencv read numpy h*w*c , caffe read n*c*h*w ( n =batch size ) channel[0] = blue
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img33[0,0,:,:] = b #batch size = 1 , so it is equal to zero, caffe is bgr too
   img33[0,1,:,:] = g
   img33[0,2,:,:] = r
   
   idx4 = line4.find(' ')
   true_age = int(line4[idx4+1:len(line4)-1])
   fullName4 = parent_dir+line4[0:idx4]
   img4 = cv2.imread(fullName4) #open image
   image = np.array(img4, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] 
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img44[0,0,:,:] = b 
   img44[0,1,:,:] = g
   img44[0,2,:,:] = r

   idx5 = line5.find(' ')
   true_age = int(line5[idx5+1:len(line5)-1])
   fullName5 = parent_dir+line5[0:idx5]
   img5 = cv2.imread(fullName5) #open image
   image = np.array(img5, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] 
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img55[0,0,:,:] = b 
   img55[0,1,:,:] = g
   img55[0,2,:,:] = r

   idx6 = line6.find(' ')
   true_age = int(line6[idx6+1:len(line6)-1])
   fullName6 = parent_dir+line6[0:idx6]
   img6 = cv2.imread(fullName6) #open image
   image = np.array(img6, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] 
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img66[0,0,:,:] = b 
   img66[0,1,:,:] = g
   img66[0,2,:,:] = r

   idx7 = line7.find(' ')
   true_age = int(line7[idx7+1:len(line7)-1])
   fullName7 = parent_dir+line7[0:idx7]
   img7 = cv2.imread(fullName7) #open image
   image = np.array(img7, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] 
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img77[0,0,:,:] = b 
   img77[0,1,:,:] = g
   img77[0,2,:,:] = r

   idx8 = line8.find(' ')
   true_age = int(line8[idx8+1:len(line8)-1])
   fullName8 = parent_dir+line8[0:idx8]
   img8 = cv2.imread(fullName8) #open image
   image = np.array(img8, dtype=np.float64)  #convert to np matrix
   b = image[:, :, 0] 
   g = image[:, :, 1] 
   r = image[:, :, 2]
   img88[0,0,:,:] = b 
   img88[0,1,:,:] = g
   img88[0,2,:,:] = r
   net1.forward( data1 = img11, data2 = img22)
   net2.forward( data1 = img11, data2 = img33)
   net3.forward( data1 = img11, data2 = img44)
   net1_m.forward( data1 = img55, data2 = img66)
   net2_m.forward( data1 = img55, data2 = img77)
   net3_m.forward( data1 = img55, data2 = img88)
   #print '......................................................................'  
   predict_age1 = 0
   predict_age2 = 0 
   predict_age3 = 0
   predict_age1_m = 0
   predict_age2_m = 0
   predict_age3_m = 0
   predict_age = 0.0   
   predict1 = net1.blobs['softmax'].data[0]
   for i in range(0,101):
      predict_age1 += i * predict1[i]

   predict2 = net2.blobs['softmax'].data[0]
   for i in range(0,101):
      predict_age2 += i * predict2[i]

   predict3 = net3.blobs['softmax'].data[0]
   for i in range(0,101):
      predict_age3 += i * predict3[i]   
   
   predict1_m = net1_m.blobs['softmax'].data[0]
   for i in range(0,101):
      predict_age1_m += i * predict1_m[i]

   predict2_m = net2_m.blobs['softmax'].data[0]
   for i in range(0,101):
      predict_age2_m += i * predict2_m[i]

   predict3_m = net3_m.blobs['softmax'].data[0]
   for i in range(0,101):
      predict_age3_m += i * predict3_m[i]
   
   predict_age1 = float(predict_age1+predict_age1_m)/2
   predict_age2 = float(predict_age2+predict_age2_m)/2
   predict_age3 = float(predict_age3+predict_age3_m)/2
   predict_age = (float(0)/2)*predict_age1 + (float(1)/2)*predict_age2 + (float(1)/2)*predict_age3

   
   diff1 = abs(float(true_age)-predict_age1)
   diff2 = abs(float(true_age)-predict_age2)
   diff3 = abs(float(true_age)-predict_age3)
   diff1_m = abs(float(true_age)-predict_age1_m)
   diff2_m = abs(float(true_age)-predict_age2_m)
   diff3_m = abs(float(true_age)-predict_age3_m)
   diff_sum1 = diff_sum1 + diff1
   diff_sum2 = diff_sum2 + diff2
   diff_sum3 = diff_sum3 + diff3
   diff_sum1_m = diff_sum1_m + diff1_m
   diff_sum2_m = diff_sum2_m + diff2_m
   diff_sum3_m = diff_sum3_m + diff3_m
   group_age = int(true_age)
   if int(true_age) <= 10:
      mae0_e += diff1
      mae0_n += diff2
      mae0_m += diff3
      num0 += 1
   elif group_age > 10 and group_age <= 20:
      mae11_e += diff1
      mae11_n += diff2
      mae11_m += diff3
      num11 += 1
   elif group_age > 20 and group_age <= 30:
      mae21_e += diff1
      mae21_n += diff2
      mae21_m += diff3
      num21 += 1
   elif group_age > 30 and group_age <= 40:
      mae31_e += diff1
      mae31_n += diff2
      mae31_m += diff3
      num31 += 1
   elif group_age > 40 and group_age <= 50:
      mae41_e += diff1
      mae41_n += diff2
      mae41_m += diff3
      num41 += 1
   elif group_age > 50 and group_age <= 60:
      mae51_e += diff1
      mae51_n += diff2
      mae51_m += diff3
      num51 += 1
   elif group_age > 60 and group_age <= 70:
      mae61_e += diff1
      mae61_n += diff2
      mae61_m += diff3
      num61 += 1
   elif group_age > 70 and group_age <= 80:
      mae71_e += diff1
      mae71_n += diff2
      mae71_m += diff3
      num71 += 1
   elif group_age > 80 and group_age <= 90:
      mae81_e += diff1
      mae81_n += diff2
      mae81_m += diff3
      num81 += 1
   elif group_age > 90 and group_age <= 100:
      mae91_e += diff1
      mae91_n += diff2
      mae91_m += diff3
      num91 += 1
   else:
      print "Error"

   diff = abs(float(true_age)-predict_age)
   diff_sum = diff_sum + diff
   #print('absoulte diff = %f' % (diff))
   print('Image name: %s, Truth age: %f, Predicted age: %f, absoulte diff: %f' % (fullName1, true_age, predict_age, diff))
   print >> fsave, 'Image name: %s, Truth age: %i, Predicted age: %f, absoulte diff: %f, predict_age1: %f, predict_age2: %f, predict_age3: %f' % (fullName1, true_age, predict_age, diff, predict_age1, predict_age2, predict_age3)
   # if c == 2:
   #    break


#mean absoult error 
MAE1= diff_sum1/count
MAE2= diff_sum2/count
MAE3= diff_sum3/count
MAE1_m= diff_sum1_m/count
MAE2_m= diff_sum2_m/count
MAE3_m= diff_sum3_m/count
MAE = diff_sum/count
print >> fsave, 'diff_sum1: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum1, count, MAE1)
print >> fsave, 'diff_sum2: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum2, count, MAE2)
print >> fsave, 'diff_sum3: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum3, count, MAE3)
print( 'diff_sum1: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum1, count, MAE1) )
print( 'diff_sum2: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum2, count, MAE2) )
print( 'diff_sum3: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum3, count, MAE3) )
print >> fsave, 'diff_sum1_m: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum1_m, count, MAE1_m)
print >> fsave, 'diff_sum2_m: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum2_m, count, MAE2_m)
print >> fsave, 'diff_sum3_m: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum3_m, count, MAE3_m)
print( 'diff_sum1_m: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum1_m, count, MAE1_m) )
print( 'diff_sum2_m: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum2_m, count, MAE2_m) )
print( 'diff_sum3_m: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum3_m, count, MAE3_m) )
print >> fsave, 'diff_sum: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum, count, MAE)
print( 'diff_sum: %f, testing sample number: %f, Mean absoult error: %f' % (diff_sum, count, MAE) ) 
# print("Group 0: %f  %f %f/ %f" % (mae0_e,mae0_n,mae0_m,num0))
# print("Group 11: %f %f %f/ %f" % (mae11_e,mae11_n,mae11_m,num11))
# print("Group 21: %f %f %f/ %f" % (mae21_e,mae21_n,mae21_m,num21))
# print("Group 31: %f %f %f/ %f" % (mae31_e,mae31_n,mae31_m,num31))
# print("Group 41: %f %f %f/ %f" % (mae41_e,mae41_n,mae41_m,num41))
# print("Group 51: %f %f %f/ %f" % (mae51_e,mae51_n,mae51_m,num51))
# print("Group 61: %f %f %f/ %f" % (mae61_e,mae61_n,mae61_m,num61))
# print("Group 71: %f %f %f/ %f" % (mae71_e,mae71_n,mae71_m,num71))
# print("Group 81: %f %f %f/ %f" % (mae81_e,mae81_n,mae81_m,num81))
# print("Group 91: %f %f %f/ %f" % (mae91_e,mae91_n,mae91_m,num91))

f1.close()
f2.close()
f3.close()
f4.close()
fsave.close()

