# Caffe implementation of MRNPE 
Based on "[Multi-Region Ensemble Convolutional Neural Networks for High-Accuracy Age Estimatios](http://www.cbsr.ia.ac.cn/users/jwan/papers/BMVC2017_age.pdf)"

The sources will include:
1. Facial Landmark Tool (.cpp file)
2. Training Files: train_net.prototxt, solver.prototxt, deploy.prototxt
3. Prediction File: predict.py
4. Trained MRNPE Caffe model (VGG version) for MORPH Album 2 dataset, pre-trained MRNPE caffe model and Raw VGG Caffe model (Google Driver Link): https://drive.google.com/open?id=0B1x8_JmGc7TFcXJrdVhOU2RDekE
5. Pytorch version codes

Unfinished (Still need to modify the instructions/codes and add some explaination for the codes.)

# Prerequisites
Caffe, C++, Matplotlib, Python 2.7

# Usage (Change code's setting of path according to users own environment)
1. Download the Morph 2 dataset. Download address: [Following Yi's protocol](http://www.cbsr.ia.ac.cn/users/dyi/agr.html)
2. data augmentation. (Rotation, Adding noises for training. Mirroring testing images for testing)
3. Run FaceLandmark/List_crop.cpp to crop the images of the sub-regions(According to Morph 2 dataset's shapeList file)
4. Train caffe net (run .sh file in the caffe net directory)
5. Prediction (python ensemble_m.py)



