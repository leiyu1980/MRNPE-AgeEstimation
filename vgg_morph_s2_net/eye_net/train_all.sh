#!/usr/bin/env sh

G_logtostderr=1 /fastdata/caffe/build/tools/caffe train \
   --solver=solver_all.prototxt \
   --weights=/data/szhou/CACD/vgg/vgg_age.caffemodel \
   --gpu=0 2>&1 |tee all.txt
