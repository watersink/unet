# This is an implementation of UNET from this paper 
 [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
the original author's code is [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## the architecture of the network from original paper
![image](https://github.com/watersink/unet/raw/master/image/network.jpg)
our changes:
1,the conv's padding is same,the paper is valid
2,the loss is the original binary_crossentropy,the paper is weighted binary_crossentropy

## what you need 
* Tensorflow>1.2
	pip install tensorflow-gpu
* Keras >= 1.0
	pip install keras
* libtiff,can be download [here](https://pypi.python.org/pypi/libtiff),also we add it in libtiff-0.4.2.tar.gz
	python2 setup.py install

## how to begin

### prepare data
the data is ISBI datasets,can be downloaded [here](http://brainiac2.mit.edu/isbi_challenge/)
 also we have added the dataset in the folder ISBI£¬
there are three .tif
	python2 split_merge_tif.py
then you will get images in data folder(30 train images and labels,30 test images)
### training
	python3 train_unet.py
finally,you will get train acc(%96.8+),val acc(%91+),val loss(0.2)
### testing
	python3 test_unet.py
<div>
<img width="400" height="400" src="https://github.com/watersink/unet/raw/master/image/000_.jpg"/>
<img width="400" height="400" src="https://github.com/watersink/unet/raw/master/image/000.jpg"/>
</div>


## references:
 [https://zhuanlan.zhihu.com/p/26659914](https://zhuanlan.zhihu.com/p/26659914)
 [https://github.com/zhixuhao/unet](https://github.com/zhixuhao/unet)