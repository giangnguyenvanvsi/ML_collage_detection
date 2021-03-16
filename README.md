# I. Introduction
The `collage_detection` is an Artifical Intelligent Project which is deveploped by AI Team from VSI.

# II. Problems & Solution

## 1. Problems
As rS engineering effort owner, I want to have an intelligent agent/service which can perform collage detection and properly mark/tag the input object (eg: picture) when the collage is detected (ask Google for more)

Some examples of collage image from RewardStyle data

![single photo](https://github.com/giangnguyenvanvsi/ML_collage_detection/blob/main/examples/00a3ce1a-7458-11eb-8026-0242ac110004.jpg)|![collage image](https://github.com/giangnguyenvanvsi/ML_collage_detection/blob/main/examples/00ae3c69-747a-11eb-8026-0242ac110004.jpg)

## 2. Solution
In the quick insight, in early work, the most commonly collage detection methods are classical computer vision. They use simple predetermined features that make them often depend on a handcrafted thresholds as parameters. They are unlearnable so not good for general dataset from RS.

Otherwise, the collage detection is very similar to splicing detection is one of the most importance problems in manipulation detection. Recently, there are several CNN-based learning techniques that can solve this similar problems effectively. 

In this repo, we implement a modified approach based on paper "`Learning Rich Features for Image Manipulation Detection`". Our approach classifies an image upload from user is collage image or not. In our approach, there is only one noise stream that leverages the noise features extracted from a SRM (steganalysis rich model) filter layer to discover the noise in-consistency between authentic and tampered regions. The features is feed and learnt by the Convolution Neural Network to get predict results.



# III. Install & Usage
## 1. Install 

Our module is build in Python 3.6 enviroment and use some machine learning frameworks as tensorflow, keras.

- Create your enviroment and install the packages listed in the requirement.txt file. 

````
pip install -r requirment.txt
````

- Download vgg16 pretrained model weight from [here](), save to weights/vgg16.ckpt

## 2. Usage
### 2.1. Trainning

````
python train.py --dataset path_to_training_data_folder
````

### 2.2. Test

```
python inference.py --input path_to_input_image
```

### 2.3. Evaluate 
About `data`, we use the common data from RS in the training process. The data includes 300 images and splited to training data and validation data with rate 0.9:0.1. Our approach also archived the accuracy 96.6% in this data. 


# IV. References
1. https://openaccess.thecvf.com/content_cvpr_2018/papers Zhou_Learning_Rich_Features_CVPR_2018_paper.pdf
2. https://github.com/LarryJiang134/Image_manipulation_detection





































































