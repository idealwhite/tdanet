
# TDANet: Text-Guided Neural Image Inpainting 
[ArXiv](https://arxiv.org/abs/2004.03212) 
<br>

This repository implements the paper "Text-Guided Neural Image Inpainting" 
by Lisai Zhang, Qingcai Chen, Baotian Hu and Shuoran Jiang. Given one masked image, the proposed 
**TDANet** generates diverse plausible results according to guidance text.

## Inpainting example

<img src='https://github.com/idealwhite/tdanet/blob/master/images/inpainting_example.png' align="center">

## Manipulation Extension example

<img src='https://github.com/idealwhite/tdanet/blob/master/images/manipulation_example.png' align="center">

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
- Download the dataset and pre-processed files.

## Datasets
- ```CUB_200```: dataset from [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html).
- ```COCO```: object detection 2014 datset from [MS COCO](https://cocodataset.org/#download).
- ```pre-processed datafiles```: train/test split, caption-image mapping, image sampling 
 and pre-trained [DAMSM](https://github.com/taoxugit/AttnGAN) weights from [GoogleDrive](https://drive.google.com/file/d/1_B7gdUwStck8Kop9hNL2YUNWF6hIxCNx/view?usp=sharing) and extarct them 
 to *dataset/* directory as specified in config.bird.yml/config.coco.yml.
## Training Demo
```
python train.py --name tda_bird  --gpu_ids 0 --model tdanet --mask_type 0 1 2 3 --img_file ./datasets/CUB_200_2011/train.flist --mask_file ./datasets/CUB_200_2011/train_mask.flist --text_config config.bird.yml
```
- **Important:** Add ```--mask_type``` in options/base_options.py for different training masks. ```--mask_file``` path is needed for **object mask**,
 ```--text_config``` refer to the yml configuration file for text setup, ```--img_file``` as the image file dir or file list.
- To view training results and loss plots, run ```python -m visdom.server``` and copy the URL [http://localhost:8097](http://localhost:8097).
- Training models will be saved under the **./checkpoints** folder.
- The more training options can be found in **./options** folder.
- **Suggestion:** use mask type 0 1 2 3 for CUB dataset and 0 1 2 4 for COCO dataset and train more than 2000 epochs. 

## Evaluation Demo
Test 
```
python test.py --name tda_bird  --img_file datasets/CUB_200_2011/test.flist --results_dir results/tda_bird  --mask_file datasets/CUB_200_2011/test_mask.flist --mask_type 3 --no_shuffle --gpu_ids 0 --nsampling 1 --no_variance
```
**Note**: remember to add  the ```--no_variance``` option to get better performance. A ```eval_tda_bird.flist``` will be generated after the test. Then in the evaluation, this file is used as the ground truth file list:

```
python evaluation.py --batch_test 60 --ground_truth_path eval_tda_bird.flist --save_path results/tda_bird
```
- Add ```--ground_truth_path``` to the dir of ground truth image path or list. ```--save_path``` as the result dir.


## Pretrained Models
Download the pre-trained models [bird inpainting](https://drive.google.com/file/d/1yGC3zPnngyrGtyWrMSYZaMXUbbiXWZGj/view?usp=sharing) or [coco inpainting](https://drive.google.com/file/d/1tqrvFFilYO3eolwqbdZYm0byQv_ahaoS/view?usp=sharing) and put them under```checkpoints/``` directory.

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

The GUI could now only avaliable in debug mode, the complete version is in development.

## TODO
- Debug the GUI application
- Further improvement on COCO quality.

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
