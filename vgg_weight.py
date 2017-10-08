import caffe
import numpy

MODEL_FILE1 = '/data/szhou/CACD/deploy.prototxt'
PRETRAINED1 = '/data/szhou/CACD/VGG_ILSVRC_16_layers.caffemodel'

MODEL_FILE2 = '/data/szhou/CACD/vgg_old/age_deploy.prototxt'
PRETRAINED2 = '/data/szhou/CACD/vgg_old/snapshot/age_iter_100.caffemodel'

net = caffe.Net(MODEL_FILE1, PRETRAINED1, caffe.TEST)
net2 = caffe.Net(MODEL_FILE2, PRETRAINED2, caffe.TEST)

conv1_1_w = net.params['conv1_1'][0].data[...]
conv1_1_b = net.params['conv1_1'][1].data[...]

net2.params['conv1_1_1'][0].data[...] = conv1_1_w
net2.params['conv1_1_1'][1].data[...] = conv1_1_b
net2.params['conv1_1_2'][0].data[...] = conv1_1_w
net2.params['conv1_1_2'][1].data[...] = conv1_1_b

conv1_2_w = net.params['conv1_2'][0].data[...]
conv1_2_b = net.params['conv1_2'][1].data[...]

net2.params['conv1_2_1'][0].data[...] = conv1_2_w
net2.params['conv1_2_1'][1].data[...] = conv1_2_b
net2.params['conv1_2_2'][0].data[...] = conv1_2_w
net2.params['conv1_2_2'][1].data[...] = conv1_2_b

conv2_1_w = net.params['conv2_1'][0].data[...]
conv2_1_b = net.params['conv2_1'][1].data[...]

net2.params['conv2_1_1'][0].data[...] = conv2_1_w
net2.params['conv2_1_1'][1].data[...] = conv2_1_b
net2.params['conv2_1_2'][0].data[...] = conv2_1_w
net2.params['conv2_1_2'][1].data[...] = conv2_1_b

conv2_2_w = net.params['conv2_2'][0].data[...]
conv2_2_b = net.params['conv2_2'][1].data[...]

net2.params['conv2_2_1'][0].data[...] = conv2_2_w
net2.params['conv2_2_1'][1].data[...] = conv2_2_b
net2.params['conv2_2_2'][0].data[...] = conv2_2_w
net2.params['conv2_2_2'][1].data[...] = conv2_2_b

conv3_1_w = net.params['conv3_1'][0].data[...]
conv3_1_b = net.params['conv3_1'][1].data[...]

net2.params['conv3_1_1'][0].data[...] = conv3_1_w
net2.params['conv3_1_1'][1].data[...] = conv3_1_b
net2.params['conv3_1_2'][0].data[...] = conv3_1_w
net2.params['conv3_1_2'][1].data[...] = conv3_1_b

conv3_2_w = net.params['conv3_2'][0].data[...]
conv3_2_b = net.params['conv3_2'][1].data[...]

net2.params['conv3_2_1'][0].data[...] = conv3_2_w
net2.params['conv3_2_1'][1].data[...] = conv3_2_b
net2.params['conv3_2_2'][0].data[...] = conv3_2_w
net2.params['conv3_2_2'][1].data[...] = conv3_2_b

conv3_3_w = net.params['conv3_3'][0].data[...]
conv3_3_b = net.params['conv3_3'][1].data[...]

net2.params['conv3_3_1'][0].data[...] = conv3_3_w
net2.params['conv3_3_1'][1].data[...] = conv3_3_b
net2.params['conv3_3_2'][0].data[...] = conv3_3_w
net2.params['conv3_3_2'][1].data[...] = conv3_3_b

conv4_1_w = net.params['conv4_1'][0].data[...]
conv4_1_b = net.params['conv4_1'][1].data[...]

net2.params['conv4_1_1'][0].data[...] = conv4_1_w
net2.params['conv4_1_1'][1].data[...] = conv4_1_b
net2.params['conv4_1_2'][0].data[...] = conv4_1_w
net2.params['conv4_1_2'][1].data[...] = conv4_1_b

conv4_2_w = net.params['conv4_2'][0].data[...]
conv4_2_b = net.params['conv4_2'][1].data[...]

net2.params['conv4_2_1'][0].data[...] = conv4_2_w
net2.params['conv4_2_1'][1].data[...] = conv4_2_b
net2.params['conv4_2_2'][0].data[...] = conv4_2_w
net2.params['conv4_2_2'][1].data[...] = conv4_2_b

conv4_3_w = net.params['conv4_3'][0].data[...]
conv4_3_b = net.params['conv4_3'][1].data[...]

net2.params['conv4_3_1'][0].data[...] = conv4_3_w
net2.params['conv4_3_1'][1].data[...] = conv4_3_b
net2.params['conv4_3_2'][0].data[...] = conv4_3_w
net2.params['conv4_3_2'][1].data[...] = conv4_3_b

conv5_1_w = net.params['conv5_1'][0].data[...]
conv5_1_b = net.params['conv5_1'][1].data[...]

net2.params['conv5_1_1'][0].data[...] = conv5_1_w
net2.params['conv5_1_1'][1].data[...] = conv5_1_b
net2.params['conv5_1_2'][0].data[...] = conv5_1_w
net2.params['conv5_1_2'][1].data[...] = conv5_1_b

conv5_2_w = net.params['conv5_2'][0].data[...]
conv5_2_b = net.params['conv5_2'][1].data[...]

net2.params['conv5_2_1'][0].data[...] = conv5_2_w
net2.params['conv5_2_1'][1].data[...] = conv5_2_b
net2.params['conv5_2_2'][0].data[...] = conv5_2_w
net2.params['conv5_2_2'][1].data[...] = conv5_2_b

conv5_3_w = net.params['conv5_3'][0].data[...]
conv5_3_b = net.params['conv5_3'][1].data[...]

net2.params['conv5_3_1'][0].data[...] = conv5_3_w
net2.params['conv5_3_1'][1].data[...] = conv5_3_b
net2.params['conv5_3_2'][0].data[...] = conv5_3_w
net2.params['conv5_3_2'][1].data[...] = conv5_3_b

fc6_w = net.params['fc6'][0].data[...]
fc6_b = net.params['fc6'][1].data[...]

print fc6_w.shape
print fc6_b.shape

fc6_w_1 = numpy.concatenate((fc6_w, fc6_w) ,axis=1)

print fc6_w_1.shape

net2.params['fc6'][0].data[...] = fc6_w_1
net2.params['fc6'][1].data[...] = fc6_b

fc7_w = net.params['fc7'][0].data[...]
fc7_b = net.params['fc7'][1].data[...]

net2.params['fc7'][0].data[...] = fc7_w
net2.params['fc7'][1].data[...] = fc7_b

net2.save('/data/szhou/CACD/vgg/vgg_age.caffemodel')

# fc8_w = net.params['fc8'][0].data[...]
# fc8_b = net.params['fc8'][1].data[...]

# print conv1_1_w.shape,conv1_1_b.shape
# print conv1_1_w.size,conv1_1_b.size
# print conv1_2_w.shape,conv1_2_b.shape

# # print pool1_w.shape,pool1_b.shape

# print conv2_1_w.shape,conv2_1_b.shape
# print conv2_2_w.shape,conv2_2_b.shape


# print conv3_1_w.shape,conv3_1_b.shape
# print conv3_2_w.shape,conv3_2_b.shape
# print conv3_3_w.shape,conv3_3_b.shape


# print conv4_1_w.shape,conv4_1_b.shape
# print conv4_2_w.shape,conv4_2_b.shape
# print conv4_3_w.shape,conv4_3_b.shape


# print conv5_1_w.shape,conv5_1_b.shape
# print conv5_2_w.shape,conv5_2_b.shape
# print conv5_3_w.shape,conv5_3_b.shape


# print fc6_w.shape,fc6_b.shape
# print fc7_w.shape,fc7_b.shape
# print fc8_w.shape,fc8_b.shape