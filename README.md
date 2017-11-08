# Cascade Residual Learning (CRL)
This repo includes the source code of the paper:
["Cascade residual learning: A two-stage convolutional neural network for stereo matching"](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Pang_Cascade_Residual_Learning_ICCV_2017_paper.pdf) by J. Pang, W. Sun, J.S. Ren, C. Yang and Q. Yan.
Please cite our paper if you find this repo useful for your work:
```
@inproceedings{pang2017cascade,
    title={Cascade residual learning: A two-stage convolutional neural network for stereo matching},
    author={Pang, Jiahao and Sun, Wenxiu and Ren, Jimmy SJ and Yang, Chengxi and Yan, Qiong},
    booktitle = {ICCV Workshop on Geometry Meets Deep Learning},
    month = {Oct},
    year = {2017}
}
```
### Prerequisites
  - MATLAB (Our scripts has been tested on MATLAB R2015a)
  - Download our trained model through this MEGA [link](https://mega.nz/#!FyhxmDLY!ZXZF2pmJEvARXCAsx85A7F4DB3juHuR2S6n6alwDTSY) or this Baiduyun [link](http://pan.baidu.com/s/1jH6tY78)
  - The KITTI Stereo 2015 dataset from the [KITTI website](http://www.cvlibs.net/download.php?file=data_scene_flow.zip)

### Testing on the KITTI dataset
  - Compile our modified Caffe and its MATLAB interface (matcaffe), our work uses the *Remap* layer (both  `remap_layer.cpp` and `remap_layer.hpp`) from the repo of ["View Synthesis by Appearance Flow"](https://github.com/tinghuiz/appearance-flow) for warping.
  - Put the downloaded model "crl.caffemodel" and the "kitti_test" folder of the KITTI dataset in the "crl-release/models/crl" folder.
  - Run `test_kitti.m` in the "crl-release/models/crl" folder for testing, our model definition `deploy_kitti.prototxt` is also in this folder.
  - Check out the newly generated folder `disp_0` and you should see the results.

### Training
We do not provide the code for training. For training, we need an in-house differentiable *interpolation* layer developed by our company, [SenseTime Group Limited](https://www.sensetime.com/). To make the code publically available, we have replaced the interpolation layer to the downsample layer of DispNet. Since the backward pass of downsample layer is not implemented, the code provided in this repo cannot be applied for training. 

On the other hand, using the downsample layer provided in DispNet does not affect the performance of the network. In fact, the network definition `deploy_kitti.prototxt` (with downsample layer) can produced a *D1-all* error of 2.67% on the [KITTI stereo 2015 leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), exactly the same as our original CRL with the in-house interpolation layer.

### Results

For your information, this is a group of results taken from the evaluation page of KITTI. To browse for more results, please click this [link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=f791987e39ecb04c1eee821ae3a0cd53d5fd28c4).

![N|Solid](http://www.cvlibs.net/datasets/kitti/results/f791987e39ecb04c1eee821ae3a0cd53d5fd28c4/image_0/000007_10.png)
![N|Solid](http://www.cvlibs.net/datasets/kitti/results/f791987e39ecb04c1eee821ae3a0cd53d5fd28c4/result_disp_img_0/000007_10.png)
![N|Solid](http://www.cvlibs.net/datasets/kitti/results/f791987e39ecb04c1eee821ae3a0cd53d5fd28c4/errors_disp_img_0/000007_10.png)