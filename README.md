# Fine-grained-Classification
The main project of Artificial Neural Network course in SYSU, solved the fine-grained visual classification task. Proudly by Tianxing Yang, Zihao Liang and Haoyu Chen.

Presentation slides used to discuss with teacher are under directory ./presentation/

## Introduction
### Step 1&2

Prepared dataloader for 2 datasets, fine-tuned on pretrained ResNet-50 model.

### Step 3

Use sahi to change annotations of two datasets to COCO format, use MMdetection framework to train the object detection model. Fine-tuning on Faster R-CNN model, using pretrained ResNet-101 as model backbone. After locating the object, we cropped it from the original picture, we done this stage by two options: one is crop the object as its original shape, and another is force the cropping area to be square. The last stage is store the cropped images and use them train and test the ResNet-50 model as Step 1-2 does before.

### Step 4

We first tried using MMagic framework to train BigGAN 256*256 model, but it came with fluctuating FID value and low-quality pictures. For this method, we use the original dataset as conditional dataset, which means we used all training set as input and marked them with corresponding labels.

Later, we switched to use DCGAN method, we used formally cropped spuare images as input. Compared to above method, we trained one model for each classes. The hyperparameters are shown in the training code. The outcome is not satisfying so we do not add them to our train set in later steps.

### Step 5

In this stage, we tried with different model of ViTs, including "vit_base_patch16_224", "vit_large_patch16_224", "vit_base_patch32_224" and "vit_small_patch16_224". Both of them are pretrained by Google on imagenet-1k dataset. We tried these model on Stanford Dogs dataset and CUB-200-2011 dataset. As we focus on the accuracy of test set, we found that the accuracy get higher at the first epoch, but soon decrease to lower level as the fine-tuning process goes. Through the accuracy rate may get moderate growth at later epoches, the accuracy rate is still lower than the first epoch.

The decrease in validation accuracy over epochs while training accuracy is increasing could be a sign of overfitting. This happens when the model learns to fit the noise in the training data too well, thus performing poorly on unseen data. It's worth noting that the Vision Transformer (ViT) model might not be the best choice for a dataset like Stanford Dogs, especially if it's a smaller dataset. ViTs generally work better on larger datasets, and convolutional neural networks (CNNs) could be more suitable for smaller datasets. So we will switch to original Resnet model for rest of stages.

### Step 6

Class Activation Mapping (CAM) is a technique that helps visualize which regions in the input image have contributed the most (or least) to the final class prediction by the Convolutional Neural Network (CNN). It essentially overlays a heat-map on top of the input image where hot spots indicate areas that have heavily influenced the prediction.

After fine-tuning the model, we extracted the last fully connected layer and transformed it into a convolutional layer with 1x1 kernel. This is because we are interested in the spatial information, which is lost when the output is passed through the fully connected layer. Then we generate output feature maps for the image, the class activation map for the most probable class is resized to the original image size.

### Step 7

The Fast Gradient Sign Method (FGSM) is a simple yet effective method for generating adversarial examples, which are then used to train the model in order to improve its robustness. We use it as attack method to test our model.


## Introduction
**IMPORTANT**: Some files list below can not be found in the GitHub repository because it is too large. We refer you download it from my server: http://43.134.189.32:12345.

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

(BigGAN)

FID value curve: ./presentation/img/pic2.png

Generated images sample: ./presentation/img/pic3.png

(DCGAN)

DCGAN code: ./legacy/task4/{bird, dogs}/dcgan.py

Results(generator and discriminator losses, scores, etc.): ./legacy/task4/{bird, dogs}/result.txt

Generated images sample: pictures under ./legacy/task4/{bird, dogs}/

### Step 5
> ViT model backbone vs. CNN backbone: explore how to effectively use ViT

(vit_base_patch16_224)

Code: ./legacy/task5/base_16_224/

Tensorboard: ./runs/task5/base_16_224/

(vit_large_patch16_224)

Code: ./legacy/task5/large_16_224/

Tensorboard: ./runs/task5/large_16_224/

(vit_base_patch32_224)

Code: ./legacy/task5/base_32_224/

Tensorboard: ./runs/task5/base_32_224/

(vit_small_patch16_224)

Code: ./legacy/task5/small_16_224/

Tensorboard: ./runs/task5/small_16_224/


### Step 6
> Interpretation of the model: visualization of model predictions

Figures: ./runs/task6/

Code: ./legacy/task6/

### Step 7
> Robustness of the model: adversarial examples as input, (optional) improve robustness

Tensorboard: ./runs/task7/

Code: ./legacy/task7/

## References
- Deep Residual Learning for Image Recognition
- ImageNet Classification with Deep Convolutional Neural Networks
- MMDetection: Open MMLab Detection Toolbox and Benchmark
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection