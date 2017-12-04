from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format
import argparse
import math
import os
import shutil
import stat
import subprocess
import sys

phase = 'TRAIN'

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

if phase == 'TEST':
  remove_old_models = False

# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "examples/Ship-Opt/Ship-Opt_trainval_lmdb"
# The database file for testing data. Created by data/VOC0712/create_data.sh
# Specify the batch sampler.
resize_width = 300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)
train_transform_param = {
        'mirror': False,
#        'mean_value': [104, 117, 123],
        }
test_transform_param = {
#        'mean_value': [104, 117, 123],
        }

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
lr_mult = 1
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.00004

# Modify the job name if you want.
job_name = "RBOX_{}_SHIPOPT_VGG".format(resize)
# The name of the model. Modify it if you want.
model_name = "RBOX_SHIPOPT_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/RBOX/Ship-Opt/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/RBOX/Ship-Opt/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/RBOX/Ship-Opt/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/Ship-Opt/results/RBOX/{}/Main".format(os.environ['HOME'], job_name)
#output_result_dir = "/var/results"
# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
name_size_file = ""
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
# Stores LabelMapItem.
label_map_file = ""

regress_size = True
regress_angle = True

# MultiBoxLoss parameters.
num_classes = 2
share_location = True
background_label_id=0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorRBox.CENTER_SIZE
ignore_cross_boundary_rbox = False
mining_type = P.MultiRBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
#prior_width = 25.0
#prior_width = prior_width / resize_width
#prior_height = 9.0
#prior_height = prior_height / resize_height
prior_widths = [45.0/300, 56.4/300, 70.9/300, 89.1/300]
prior_heights = [9.0/300, 11.3/300, 14.2/300, 17.8/300]
prior_width = -1;
prior_height = -1;
#prior_widths = [prior_width]
#prior_heights = [prior_height]
multirbox_loss_param = {
    'loc_loss_type': P.MultiRBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiRBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiRBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_rbox': ignore_cross_boundary_rbox,
    'prior_width': prior_width,
    'prior_height': prior_height,
    'regress_size': regress_size,
    'regress_angle': regress_angle,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating priors.
# minimum dimension of input image
min_dim = 300
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
mbox_source_layers = ['conv4_3']
steps = [8]
# L2 normalize conv5_3.
normalizations = [20]
#normalizations = [-1]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorRBox.CENTER_SIZE:
  #prior_variance = [0.1, 0.1, 0.1]
  prior_variance = [0.1, 0.1, 0.2, 0.2, 0.1] # modified by LL
else:
  prior_variance = [0.1]
rotate_angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
flip = False
clip = False

# Solver parameters.
# Defining which GPUs to use.
# gpus = "0,1,2,3"
# gpulist = gpus.split(",")
# num_gpus = len(gpulist)
gpus = "0,1"
gpulist = gpus.split(",")
num_gpus = len(gpulist)
#num_gpus = 0;

# Divide the mini-batch to different GPUs.
batch_size = 16
accum_batch_size = 16
test_batch_size = 64
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])
if normalization_mode == P.Loss.NONE:
  base_lr /= batch_size_per_device
elif normalization_mode == P.Loss.VALID:
  base_lr *= 25. / loc_weight
elif normalization_mode == P.Loss.FULL:
  # Roughly there are 2000 prior bboxes per image.
  # TODO(weiliu89): Estimate the exact # of priors.
  base_lr *= 2000.


solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [100000, 130000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 300000,
    'snapshot': 10000,
    'display': 20,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
#    'test_iter': [test_iter],
#    'test_interval': 1,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }


### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_data)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
print("Create train net")
net.data, net.label = CreateAnnotatedRDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, name="data")
#net.data2, net.label2 = CreateAnnotatedRDataLayer(train_data2, batch_size=batch2_size_per_device,
#        train=True, output_label=True, label_map_file=label_map_file,
#        transform_param=train_transform_param, name="data2")

#name = 'data'
#data_layers = [net.data1, net.data2]
#net[name] = L.Concat(*data_layers, axis=0)

#net.slience1 = L.Silence(net.label2, ntop=0,
#    include=dict(phase=caffe_pb2.Phase.Value('TRAIN')))

print("Create train net done")

VGGNetBodyCut(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)
#net.slience = L.Silence(net.fc7, ntop=0,
#    include=dict(phase=caffe_pb2.Phase.Value('TRAIN')))
#AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)
    
mbox_layers = CreateMultiRBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm,
        steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult,
        rotate_angles=rotate_angles, 
        prior_widths = prior_widths, prior_heights = prior_heights,
        regress_size = regress_size, regress_angle = regress_angle);

# Create the MultiBoxLossLayer.
name = "mbox_loss_plane"
mbox_layers.append(net.label)
net[name] = L.MultiRBoxLoss(*mbox_layers, multirbox_loss_param=multirbox_loss_param,
        loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

#Add OutputData Layer
name = "output_data"
# net[name] = L.OutputData(net['mbox_conf'])
#
# AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)


with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)


# Create test net.
net = caffe.NetSpec()
net.data = L.DummyData(shape=dict(dim=[test_batch_size,3,300,300]))

VGGNetBodyCut(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)
    
#AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

mbox_layers = CreateMultiRBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, 
	    steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult, rotate_angles=rotate_angles,
        prior_widths = prior_widths, prior_heights = prior_heights,
        regress_size = regress_size, regress_angle = regress_angle)

conf_name = "mbox_conf_plane"
if multirbox_loss_param["conf_loss_type"] == P.MultiRBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multirbox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]


with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

shutil.copy(test_net_file, job_dir)

# Create deploy net.
# Remove the first and last layer from test net.

#net.slience = L.Silence(net.fc7, ntop=0,
#    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    # del net_param.layer[0]
    # del net_param.layer[-1]
    #net_param.name = '{}_deploy'.format(model_name)
    #net_param.input.extend(['data'])
    #net_param.input_shape.extend([
    #    caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

shutil.copy(deploy_net_file, job_dir)


# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        #test_net=[deploy_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)
print("Create solver done")

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
if phase == 'TEST':
  pretrain_model = "{}_iter_{}.caffemodel".format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
if phase == 'TRAIN':
  with open(job_file, 'w') as f:
    f.write('cd {}\n'.format(caffe_root))
    f.write('./build/tools/caffe train \\\n')
    f.write('--solver="{}" \\\n'.format(solver_file))
    f.write(train_src_param)
    if solver_param['solver_mode'] == P.Solver.GPU:
      f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
    else:
      f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)
# Run the job.
print("Run the job")
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
