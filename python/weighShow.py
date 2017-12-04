# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:14:30 2017

@author: cgw
"""

import numpy as np
#import matplotlib.pyplot as plt

conv2_param_file = 'conv3_3_params.txt'
conv2_params = np.loadtxt(conv2_param_file)
conv2_params = conv2_params.reshape(-1,9)

temp = conv2_params[0:256]
conv2_params_0 = temp.T
u,sigma,v = np.linalg.svd(conv2_params_0)
print sigma
#plt.figure()
#plt.plot(sigma)
