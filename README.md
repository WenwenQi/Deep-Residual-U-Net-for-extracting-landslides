# Deep-Residual-U-Net-for-extracting-landslides
> This work has been submitted to the journal 'Landslides'. To use this work, please contact and get a permission from Wenwen Qi (qiww@lreis.ac.cn). 

> The Deep Residual U-Net is proposed based on the U-Net and the residual neural network (ResNet).<br>
> This work introduced a new deep learning approach designed to automatically identify landslides from very high spatial resolution (0.5 m) images. This proposed method was tested in Tianshui city, Gansu province, where a heavy rainfall triggered more than 10,000 landslides in August 2013. The method only used 3-band and achieved high performance (recall 88.5%, precision 63.5%) in this spatially heterogeneous region. The authors wish the landslide community could use this state-of-the-art method to aid landslide mappings. <br>

## Dataset:
GeoEye-1, 0.5m resolution, Bands [4,3,2].<br>
landslides train and test data：<br>
(1) GeoEye-1 image tiles, 600*600 pixels,  1443 tiles;<br>
(2) landslides map label tiles, 600 600 pixels, 1443 tiles;<br>
you can get this dataset  from Email: qiww@lreis.ac.cn .<br>
![](https://github.com/WenwenQi/Deep-Residual-U-Net-for-extracting-landslides/blob/master/data-ls/data%20samples/train_282.jpg "image")
![](https://github.com/WenwenQi/Deep-Residual-U-Net-for-extracting-landslides/blob/master/data-ls/data%20samples/train_282_label.jpg "groundtruth")

## Output
![](https://github.com/WenwenQi/Deep-Residual-U-Net-for-extracting-landslides/blob/master/data-ls/lanslide.gif "output result")
> NIR R G image > [Red] groundtruth > [Yellow] extract result > overlay result <br>
## Requirements:
> Ubuntu<br>
> Python3<br>
> pyTorch 0.4<br>
> or based on NVIDIA-docker image 'pyTorch': nvcr.io/nvidia/pytorch:18.08-py3<br>
## Citation
> Use this bibtex to cite this repository:<br>
> Qi, W., Wei, M., Yang, W., Xu, C., & Ma, C. (2020). Automatic Mapping of Landslides by the ResU-Net. Remote Sensing, 12(15), 2487. https://www.mdpi.com/2072-4292/12/15/2487 <br>
> @misc{wenwenqi_reunet_2019,<br>
>  title={Regional landslides mapping by Deep Residual U-Net},<br>
>  author={Wenwen Qi},<br>
>  year={2019},<br>
>  publisher={Github},<br>
>  journal={GitHub repository},<br>
>  howpublished={\url{https://github.com/WenwenQi/Deep-Residual-U-Net-for-extracting-landslides}},<br>
>}<br>
