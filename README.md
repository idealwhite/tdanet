
# TDANet: Text-Guided Neural Image Inpainting 
[ArXiv](https://arxiv.org/abs/2004.03212) 
<br>

This repository implements the training, testing and editing tools for "Text-Guided Neural Image Inpainting" 
by Lisai Zhang, Qingcai Chen, Baotian Hu and Shuoran Jiang. Given one masked image, the proposed 
**TDANet** is able to generate diverse plausible results according to guidance text.

## Inpainting example

<img src='https://github.com/idealwhite/tdanet/raw/master/images/inpainting_example.png' align="center">

## Manipulation Extension example

<img src='https://github.com/idealwhite/tdanet/raw/master/images/manipulation_example.png' align="center">

# Getting started
## Installation
This code was tested with Pytoch 1.2.0, CUDA 10.1, Python 3.6 and Ubuntu 16.04

- Install Pytoch 1.2.0, torchvision, and other dependencies from [http://pytorch.org](http://pytorch.org)
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate) for visualization

```
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/idealwhite/tdanet
cd tdanet
```
- Download the pre-trained DAMSM model in [AttnGAN](https://github.com/taoxugit/AttnGAN) and save to *dataset/*
 directory specified in config.bird.yml/config.coco.yml file.

## Datasets
- ```CUB_200``` original training images from [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html).
- ```COCO``` original training images from [MS COCO](https://cocodataset.org/#download).

## Training
```
python train.py --name tda_bird  --gpu_ids 0 --model tdanet
```
- Add ```--mask_type``` in options/base_options.py for different training masks. ```--mask_file``` path is needed for **object mask**,
 ```--text_config``` refer to the yml configuration file for text setup, ```--img_file``` as the image file dir or file list.
- To view training results and loss plots, run ```python -m visdom.server``` and copy the URL [http://localhost:8097](http://localhost:8097).
- Training models will be saved under the **checkpoints** folder.
- The more training options can be found in **options** folder.

## Testing
```
python test.py --name tda_bird --no_shuffle --gpu_ids 1 --nsampling 1 --no_variance
```
- Add ```--mask_type``` in options/base_options.py test various masks. 
```--mask_file``` path to json file is needed for *object mask*,
 ```--text_config``` refer to the yml configuration file for text setup, 
 ```--img_file``` as the test image file dir or file list.
 set ```--no_variance``` to sample from the center of latent distribution.
- The default results will be saved under the *results* folder. Set ```--results_dir``` for a new path to save the result.

## Evaluation

```
python evaluation.py --batch_test 60
```
- Add ```--ground_truth_path``` to the dir of ground truth image path or list. ```--save_path``` as the result dir.


## Pretrained Models
Download the pre-trained models using the following links and put them under```checkpoints/``` directory.
- ```bird inpainting```: Comming soon. 
- ```coco inpainting```: Comming soon.

The pre-trained model is preparing.

## GUI

- Install the [PyQt5](https://pypi.org/project/PyQt5/) for GUI operation

```
pip install PyQt5
```

Basic usage is:
```
python -m visdom.server
python ui_main.py
```

The complete version is in development.

## Next
- Improvement on COCO quality

## License
This software is for educational and academic research purpose only. If you wish to obtain a commercial royalty bearing license to
 this software, please contact us at lisaizhang@foxmail.com.

## Acknowledge
We would like to thanks Zheng et al. for providing their source code. This project is fit from their greate [Pluralistic Image Completion Project](https://github.com/lyndonzheng/Pluralistic-Inpainting).

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{tdanet,
  title={Text-Guided Neural Image Inpainting},
  author={Lisai Zhang, Qingcai Chen, Baotian Hu, Shuoran Jiang},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia (MM '20)},
  year={2020}
}
```
