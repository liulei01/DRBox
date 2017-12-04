# DRBox
By Lei Liu (mail: lliu1@mail.ie.ac.cn)

### Introduction
DRBox is used for detection tasks where the objects are orientated arbitrarily. This code show examples that DRBox is used to detect vehicles, ships and airplanes in remote sensing images. I'm also looking forward for its use in other problems.

The codes are modified from the original Caffe and [SSD](https://github.com/weiliu89/caffe/tree/ssd). 

### Citing DRBox

The article for this method can be downloaded here: [arXiv:1711.09405](https://arxiv.org/abs/1711.09405). Please cite this work in your publications if it helps your research.

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train](#train)
(4. [Deployment](#deployment)

### Installation
1. DRBox is written in Caffe with some newly defined layers. So you should prepare nessasary environment for Caffe installation.
DThen you can get the code:
```Shell
git clone https://github.come/liulei01/drbox.git
```
2. Matlab is also neccessary so that the results can be viewed.

3. If you only want to apply our trained models directly to your applications, then you can ignore the following instruction and jump to [Deployment](#deployment).

4. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
```Shell
# Modify Makefile.config according to your Caffe installation.
cp Makefile.config.example Makefile.config
make -j8
# Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
make py
```

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`

2. Download the training data for DRBox at *** (The training data will be available soon) , extract them at data/.

3. Run create_data.sh in each subfolders in data/ to create LMDB for training. For example, when you want to train a airplane detection network, then you can 
```Shell
cd $CAFFEROOT
./data/Airplane/create_data.sh
```

### Train
DRBox is now designed as a single task network. So you should train it for each type of objects separately. The python codes are in examples/rbox/. If you want to train a airplane detection network, then you can start training by:
```Shel
cd $CAFFEROOT
python examples/rbox/rbox_pascal_airplane.py
```

Training for vehicle is similar with airplane.
```Shell
cd $CAFFEROOT
python examples/rbox/rbox_pascal_car.py
```

Before training for ship, you should replace src/caffe/util/rbox_util.cpp with src/caffe/util.rbox_util.cpp.ship and rebuilding the codes. The reason is that we ignore the head and tail of a ship to make the problem easier.
```Shell
cd $CAFFEROOT
mv src/caffe/util/rbox_util.cpp src/caffe/util/rbox_util.cpp.old
mv src/caffe/util/rbox_util.cpp.ship src/caffe/util/rbox_util.cpp
make -j8
python examples/rbox/rbox_pascal_ship_opt.py
```

The trained models are stored in models/RBOX/.

### Deployment
The codes for deployment are in examples/rbox/deploy. 

1. If you only want to apply a pre-trained models directly to your applications in any Caffe environment, then you can copy this folder to your own Caffe folder and run the following commands.
```Shell
mv librbox.cpp.code librbox.cpp
g++ -o librbox.so -shared -fPIC librbox.cpp
cp deploy.py.general_example deploy.py
# you should modify the following file accordingly.
python deploy.py
```

2. Otherwise, make sure that caffemodel file, deploy.prototxt file are generated during training. Then run the following commands:
```Shell
mv librbox.cpp.code librbox.cpp
g++ -o librbox.so -shared -fPIC librbox.cpp
mv librbox.cpp librbox.cpp.code
cp deploy.py.example deploy.py
# you should modify the following file accordingly.
python deploy.py
```

### View Results
The detection results are stored in a text file named like output.rbox.score. We provide a matlab function to view the results. In matlab, open examples/rbox/deploy/SelectRotatedTarget.m and run it. You are asked to select the demo tiff figure and the output.rbox.score file, then the results will be plotted. Press Z to zoom in and X to zoom out. In the first view, each result is plotted in a red circle, you can press Z to change them to rectangles. 
