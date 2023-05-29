# Fine-grained-Classification
The main project of Artificial Neural Network course in SYSU, solved the fine-grained visual classification task. Proudly by Tianxing Yang, Zihao Liang and Haoyu Chen.

## Introduction
### Step 1&2

Prepared dataloader for 2 datasets, fine-tuned on pretrained ResNet-50 model.

### Step 3

Use sahi to change annotations of two datasets to COCO format, use MMdetection framework to train the object detection model. Fine-tuning on Faster R-CNN model, using pretrained ResNet-101 as model backbone. After locating the object, we cropped it from the original picture, we done this stage by two options: one is crop the object as its original shape, and another is force the cropping area to be square. The last stage is store the cropped images and use them train and test the ResNet-50 model as Step 1-2 does before.

### Step 4

(to be completed ...)


## Introduction

**IMPORTANT**: Some files list below can not be found in the GitHub repository because it is too large. We refer you download it from 43.134.189.32:12345.

### Step 1&2
> Various training tricks to improve model performance

> Transfer learning: fine-tune pretrained model

A unified code for all dataset: ./dataloader.py

**Stanford Dogs dataset**

Main Code: ./legacy/task1-2/dog.py

Result: Tensorboard under directory ./runs/task1-2/dog/

**CUB-200-2011 dataset**

Main Code: ./legacy/task1-2/bird.py

Result: Tensorboard under directory ./runs/task1-2/bird/

### Step 3
> Attend to local regions: object localization or segmentation

**Stanford Dogs dataset**

Annotation convert code: ./legacy/task3/StanfordDogs_convert.py

MMdetection training config(please refer to document of MMDetection for detailed meaning): ./MMDetection/StanfordDogs_train.py

Object Detection Result: Tensorboard under directory ./runs/task3/dog/objectdetection, more images can be found under ./runs/task3/dog/objectdetection/vis_image

Original Dataset Pictures: ./data/StanfordDogs/Images

Crop Dataset code: ./legacy/task3/StanfordDogs_crop.py

Cropped Dataset Pictures: ./data/StanfordDogs/croppedImages

Square Crop Dataset code: ./legacy/task3/StanfordDogs_squareCrop.py

Square Cropped Dataset Pictures: ./data/StanfordDogs/squareCroppedImages

Train & Set result for cropped pictures: Tensorboard under directory ./runs/task3/dog/cropClassification

Train & Set result for square cropped pictures: Tensorboard under directory ./runs/task3/dog/squareCropClassification

**CUB-200-2011 dataset**

Annotation convert code: ./legacy/task3/CUB-200-2011_convert.py

MMdetection training config(please refer to document of MMDetection for detailed meaning): ./MMDetection/CUB-200-2011_train.py

Object Detection Result: Tensorboard under directory ./runs/task3/bird/objectdetection, more images can be found under ./runs/task3/bird/objectdetection/vis_image

Original Dataset Pictures: ./data/CUB-200-2011/CUB_200_2011/images

Crop Dataset code: ./legacy/task3/CUB-200-2011_crop.py

Cropped Dataset Pictures: ./data/CUB-200-2011/CUB_200_2011/croppedimages

Square Crop Dataset code: ./legacy/task3/CUB-200-2011_squareCrop.py

Square Cropped Dataset Pictures: ./data/CUB-200-2011/CUB_200_2011/squarecroppedimages

Train & Set result for cropped pictures: Tensorboard under directory ./runs/task3/bird/cropClassification

Train & Set result for square cropped pictures: Tensorboard under directory ./runs/task3/bird/squareCropClassification

### Step 4
> Synthetic image generation as part of data augmentation

(to be completed ...)

## Install Environment

(to be completed ...)

## References
- Deep Residual Learning for Image Recognition
- ImageNet Classification with Deep Convolutional Neural Networks
- MMDetection: Open MMLab Detection Toolbox and Benchmark
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection


