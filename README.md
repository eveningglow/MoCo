# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Unofficial pytorch implementation of __Momentum Contrast for Unsupervised Visual Representation Learning__ ([Paper](https://arxiv.org/abs/1911.05722)).  

## Requirements  
- PyTorch 1.4.0
- torchvision 0.5.0
- PIL 7.0.0
- matplotlib 2.0.2
- PyYAML 3.12 (Optional)  

## Dataset  
### ImageNet  
Download the dataset and untar. It will create subdirectories for each class with images belonging to that class.   
``` 
cd YOUR_ROOT  
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar  
tar xvf ILSVRC2012_img_train.tar  
```  
As a result, the subdirectories for training dataset will be located in ```YOUR_ROOT/ILSVRC/Data/CLS-LOC/train```.

### STL-10  
One can download this dataset from [here](http://ai.stanford.edu/~acoates/stl10/), or just use [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html). This repository handles STL-10 dataset via torchvision. Please check ```dataloader.py``` for details.

## Hardware  
All results in this repository are produced with __6 NVIDIA TITAN Xp GPUs__. To produce the best performance, multi-gpu is necessary.
  
## Training  
You can train Resnet-50 encoder in self-supervised manner with ImageNet dataset by running the command below.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --dataset_root=YOUR_ROOT/ILSVRC/Data/CLS-LOC/train --shuffle_bn --save_config 
```
The training output such as loss graph and weight of the encoder will be saved in ```MoCo/output/IMAGENET-64/v1```. You can change this location by changing the arguments ```--output_root```, ```--dataset_name```, and ```--exp_version```.

## Test  
You can evaluate the pretrained encoder with STL-10 dataset by running the command below.
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_root=YOUR_ROOT/STL-10 --load_epoch=100
```
This command will train a linear feature classifier that takes the feature vectors from pretrained encoder as inputs with STL-10 dataset.
