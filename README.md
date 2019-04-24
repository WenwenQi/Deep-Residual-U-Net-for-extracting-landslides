# Deep-Residual-U-Net-for-extracting-landslides
> The Deep Residual U-Net is proposed based on the U-Net and the residual neural network (ResNet).<br>
> This work introduced a new deep learning approach designed to automatically identify landslides from very high spatial resolution (0.5 m) images. This proposed method was tested in Tianshui city, Gansu province, where a heavy rainfall triggered more than 10,000 landslides in August 2013. The method only used 3-band and achieved high performance (recall 88.5%, precision 63.5%) in this spatially heterogeneous region. The authors wish the landslide community could use this state-of-the-art method to aid landslide mappings. <br>
## Dataset:
landslides train and test data：<br>
(1) GeoEye-1 image tiles, 600 600,  1443 tiles;<br>
(2) landslides map label tiles, 600 600, 1443 tiles;<br>
you can get this dataset  from Email: qiww@lreis.ac.cn .<br>
![](https://github.com/WenwenQi/Deep-Residual-U-Net-for-extracting-landslides/blob/master/data-ls/data%20samples/train_278.jpg)
![](https://github.com/WenwenQi/Deep-Residual-U-Net-for-extracting-landslides/blob/master/data-ls/data%20samples/train_278_label.jpg)

## Requirements:
> Ubuntu<br>
> Python3<br>
> pyTorch 0.4<br>
> or based on NVIDIA-docker image 'pyTorch': nvcr.io/nvidia/pytorch:18.08-py3<br>

