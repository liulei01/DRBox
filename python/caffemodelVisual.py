# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:03:51 2017

@author: cgw
"""

import caffe
import numpy as np

np.set_printoptions(threshold='nan')

model_file = 'models/RBOX/BEIJING/RBOX_300x300_25X9/train.prototxt'
caffemodel_file = 'models/RBOX/BEIJING/RBOX_300x300_25X9/RBOX_BEIJING_RBOX_300x300_25X9_iter_78647.caffemodel'

net = caffe.Net(model_file, caffemodel_file, caffe.TRAIN)

conv2_1_param = 'conv3_3_params.txt'
pf = open(conv2_1_param,'w')
weight = net.params['conv3_3'][0].data
weight_shape = net.params['conv4_1'][0].data.shape
h = weight_shape[-2]
w = weight_shape[-1]
for cin in range(len(weight[0])):
    for cout in range(len(weight)):
        for k in range(h):
            for l in range(w):
                pf.write('%f ' %weight[cout][cin][k][l])
            pf.write('\n')
        pf.write('\n')
    pf.write('\n')
pf.close
