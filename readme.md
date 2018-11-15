# LF-Net: Learning Local Features from Images

This repository is a tensorflow implementation  for Y.  Ono, E. Trulls, P. Fua,
K.M. Yi, "LF-Net: Learning Local Features from Images". If you use this code in
your research, please cite [the paper](https://arxiv.org/abs/1805.09662). 


![comparison](/teasers/teasers.png)

## Installation

This code is based on Python3 and tensorflow with CUDA-9.0. For more details on
the required  libraries, see  `requirements.txt`. You  can also  easily prepare
this by doing

```
pip install -r requirements.txt
```

## Pretrained models and example dataset

Download                             the                            [pretrained
models](http://webhome.cs.uvic.ca/~kyi/files/2018/lf-net/pretrained.tar.gz) and
the                                                                [scare_coeur
sequence](http://webhome.cs.uvic.ca/~kyi/files/2018/lf-net/sacre_coeur.tar.gz). Extract
them to the current folder so that they fall under `release/models/outdoor` for
example.

For other datasets, we do not plan to release them at the moment. Please do not
contact us for  explanations on the training phase. We  are providing them **as
is** as a reference implementation.

### Updates since the arXiv version

The provided pre-trained  models are trained with full  360 degree augmentation
for  orientation. Thus,  the results  you get  from these  models are  slightly
different  from  the  one  reported  in  arXiv.  We  have  further  included  a
consistency term on the orientation assignment.

## Running the keypoint extraction demo

To run LF-Net for all images in a given directory, simply type:

```
python run_lfnet.py --in_dir=images --out_dir=outputs
```

## Running 2-view matching demo with OpenCV
To run LF-Net for 2-view matching, simple type:

```
python test_one_match.py --train=test_1/daytime2.jpg --query=test_1/night2.jpg --out_dir=outputs
```

In addition, you can easily do the 2-view matching demo through
`notebooks/demo.ipynb` .

## Training

Training code can be found in `train_lfnet.py`. We will **not** provide any
support for the training process and datasets. All issues related to this topic
will be closed without answers.


## Some Examples

| Outdoor dataset</br> Top: LF-Net, Bottom: SIFT | Indoor dataset </br>Top: LF-Net, Bottom: SIFT | Webcam dataset</br>Top: LF-Net, Bottom: SIFT |
|:---------|:--------------------|:----------------|
| ![outdoor](/teasers/sfm_ours_sift.gif)     | ![indoor](/teasers/scannet_ours_sift.gif) | ![webcam](/teasers/webcam_ours_sift.gif) |


