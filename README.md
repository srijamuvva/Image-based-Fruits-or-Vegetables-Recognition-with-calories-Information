# Machine-Learning-Image-based-Fruits-and-Vegetables-Recognition-with-calories-
 ```
i
```
# ABSTRACT

This project introduces a lightweight deep learning model for fruit and vegetable recognition,
combining a modified MobileNetV2 architecture with an attention module. The model begins by
extracting convolutional features to capture essential object-based information, followed by an
attention mechanism that enhances semantic understanding. By integrating these modules, the
model effectively balances high-level object representation with nuanced semantic details crucial
for accurate classification.
Utilizing transfer learning from a pre-trained MobileNetV2 model optimizes training efficiency
and adaptation to specific fruit datasets. Evaluation across three benchmark datasets demonstrates
that the proposed model surpasses four recent deep learning methods in classification accuracy
while significantly reducing the number of trainable parameters. This makes it well-suited for
applications in industries related to fruit cultivation, retail, and processing, where automatic fruit
identification are valuable for operational efficiency and quality control.


```
ii
```
# LIST OF FIGURES

S.No. Fig No. Fig Name Page No.

1. 3.2.1 Architecture Diagram 5
2. 3.2.2 Workflow Diagram 6
3. 4.1.1 Dataset 7
4. 4.2.1 Code 7
5. 4.3.1 Outputs 8


```
iii
```
# LIST OF TABLES

S.No. Table No. Table Name Page No.

1. 2.1 Literature Survey 3
2. 4.4.1 Accuracy Results 9


## 1. INTRODUCTION

In the modern world, technology plays a pivotal role in promoting healthy eating habits.

With the increasing awareness of nutritional intake, there is a growing demand for

applications that can assist individuals in identifying and understanding the nutritional

value of the foods they consume. This project, titled "Image-Based Fruits or Vegetables

Recognition with Calories", aims to address this need by leveraging machine learning

and computer vision technologies.

This project focuses on the classification of fruits and vegetables. It is a user-friendly

web application designed for simplicity and accessibility. Users are required to upload

an image of any fruit or vegetable, and our system will automatically classify the image,

providing the predicted name of the object. Additionally, we have integrated a module

that calculates the calories of the identified item. Being a web-based application, it is

easily accessible via any standard web browser, ensuring seamless usability for all

users.

1.1 Objectives

```
 Develop an Automated Image Classification System.
 Create a system that allows users to upload images of fruits and vegetables for
automatic classification.
 Integrate a feature that provides the calorie content of the identified fruit or
vegetable.
 Utilize deep learning techniques and web scraping to gather comprehensive data
and improve the accuracy of the recognition system.
 Showcase the practical application of deep learning in everyday life by
developing a user-friendly web interface for the recognition system.
```
1.2 Methodology

To classify fruits and vegetables, a large collection of the fruits and vegetables images

are required. The images are downloaded from kaggle. In this section the methodology

followed is discussed in detail.


1.2.1 Dataset

Introduction

In this project we are using the “Fruit and Vegetable Image Recognition” dataset. This

dataset have 36 classes, and almost 100 images for each class so we can say we have

3600+ training images. We have 10 images for each category in Train/Validation.

What you will learn:

```
 How to use Python and TensorFlow to train an image classifier
 How to classify images with the trained classifier
```
1.2.2 Preprocessing

# val_images generator is used to provide validation data during training. It is similar to

# train_images but does not apply augmentation. File paths and labels are combined into

a DataFrame. The DataFrame is shuffled to ensure randomness.

1.2.3 Training

A pretrained MobileNetV2 model is used as the base model without its top layers

# (include_top=False). The model is initialized with weights trained on ImageNet.

Custom dense layers are added on top of the pretrained model to adapt it to the specific

classification task. The final layer uses a softmax activation function to output

probabilities for each of the 36 classes.

1.2.4 Validation: A separate validation set is used during training to tune the model’s

hyperparameters and check for overfitting. The validation data is augmented similarly

to the training data to ensure consistency.

1.2.5 Testing

The test set is used to evaluate the final performance of the trained model. Unlike

training and validation, the test set does not include data augmentation (except for

standard preprocessing) to simulate real-world data


## 2. LITERATURE SURVEY

```
Table 2.1 Literature Survey
```
S.No. Title Summary

[1] (^) Fruit Image Classification
Model Based on
MobileNetV2 with Deep
Transfer Learning
Technique
The rise of AI has boosted the use of smart imaging devices and deep
learning models like CNN for image classification, which don't need
handcrafted features. In horticulture, fruit classification demands
expert knowledge. An automated system using CNN can classify
fruits without human effort, addressing this industry challenge.

### [2]

```
Fruit classification using
attention-based
MobileNetV2 for
industrial applications
```
```
Recent deep learning methods for fruit classification often require
heavy-weight architectures. This paper proposes a lightweight model
using MobileNetV2 and an attention module, combining high-level
object-based and semantic information. Evaluations on three
benchmark datasets show superior accuracy and fewer trainable
parameters, making it suitable for industry adoption.
```
### [3]

```
Fruit and Vegetable
Classification Using
MobileNetV2 Transfer
Learning Model
```
```
This research explores the use of the MobileNetV2 model for
classifying fruits and vegetables into multiple categories,
demonstrating high accuracy and efficiency. Utilizing a substantial
dataset, the model achieved impressive results, indicating its
potential to revolutionize agricultural monitoring, quality assurance,
and retail automation. The study highlights the benefits of improved
classification accuracy for reducing waste, enhancing product
quality, and supporting sustainable practices in the food and
agriculture sectors.
```
[4] Recognition of food type
and calorie estimation
using neural network

```
As obesity becomes a major health issue, controlling diet through
low-calorie, high-nutrition foods is essential. This study uses a
multilayer perceptron model to identify food types and estimate
calorific values, aiding in healthy food choices. Implemented in
MATLAB, the method showed acceptable accuracy in detecting food
```

```
items and estimating calories. acceptable accuracy in detecting food
items and estimating calories
```
[5] Classification of Fruits
using Deep Learning
Algorithms

```
Fruit classification is vital for agriculture, especially for import and
export. Using Gaussian filters for pre-processing, fruits like apples,
oranges, and bananas are classified and assessed for quality.
Convolutional Neural Networks, AlexNet, and MobileNetV2 were
used, with MobileNetV2 achieving the highest accuracy in both fruit
type and defect classification.
```

## 3. DESIGN

3.1 Introduction

We are using MobilenetV2 architecture. MobileNetV2 is a convolutional neural

network architecture that seeks to perform well on mobile devices. It is based on an

inverted residual structure where the residual connections are between the bottleneck

layers. MobilenetV2 supports any input size greater than 32 x 32.

```
 In MobileNetV2, there are two types of blocks. One is residual block with stride
of another one is block with stride of 2 for downsizing.
```
```
 There are 3 layers for both types of blocks.
```
```
 This time, the first layer is 1×1 convolution with ReLU6.
```
```
 The second layer is the depth wise convolution.
```
```
 The third layer is another 1×1 convolution but without any non-linearity. It is
claimed that if RELU is used again, the deep networks only have the power of
a linear classifier on the non-zero volume part of the output domain.
```
3.2 Architecture

```
Fig 3.2.1 MobileNetV2 Architecture
```

3.3 Workflow

In this we are going to see how our web-application is working. We have divided our

modules, so our task is going to be easy. Our frontend-backend will be handled by the

Streamlit. User will visit our application by URL. An upload button is available for user

to upload the image. After the uploading the image, our system will do the task

automatically.

```
 User will upload the image. That image will be stored into the local system.
 Now pillow will resize the image according to our model shape, it will convert
into vector.
 Now this vector will be passed to our model, our model will classify the class
of category.
 We will get the ID of category, now we need to map the labels according to the
ID.
 Now our system will web-scrap the calories for predicted object. Our
application will display the Result and Calories into our application.
```
```
Fig 3.3.1 Workflow Diagram
```

## 4. IMPLEMENTATION

4.1 Dataset

```
Fig 4.1.1 Dataset
```

4.2 Coding

```
Fig 4.2.1 Code
```
4.3 OUTPUTS


### `


### 4.4 ACCURACY TABLE WITH METRICS

```
Table 4.4.1 Accuracy Results
MODEL ACCURACY (%)
MobileNetV2 96
```

## 5. CONCLUSION AND FUTURE SCOPE

In conclusion, the fruit and vegetable recognition and calorie counter project

successfully developed a system that accurately identifies various fruits and vegetables

from images and provides corresponding calorie information.

This approach simplifies dietary tracking and calorie counting, making it easier for

users to make informed nutritional choices. Leveraging advanced machine learning

techniques, the project has the potential to promote healthier eating habits.

Potential extensions like recognizing more categories , real time recognition and

integration with mobile applications

Broader applications in industries such as smart grocery stores, automated sorting in

agriculture and dietary monitoring.


## 6. REFERENCES

[1] Gulzar, Y., 2023. Fruit image classification model based on MobileNetV2 with deep

transfer learning technique. Sustainability, 15 (3), p.1906.

[2] Shahi, T.B., Sitaula, C., Neupane, A. and Guo, W., 2022. Fruit classification using

attention-based MobileNetV2 for industrial applications. Plos one, 17 (2), p.e0264586.

[3] Kaur, G., Sharma, N., Chauhan, R., Pokhariya, H.S. and Gupta, R., 2023, December.

Fruit and Vegetable Classification Using MobileNet V2 Transfer Learning Model.

In 2023 3rd International Conference on Smart Generation Computing,

Communication and Networking (SMART GENCON) (pp. 1-6). IEEE.

[4] Kumar, R.D., Julie, E.G., Robinson, Y.H., Vimal, S. and Seo, S., 2021. Recognition

of food type and calorie estimation using neural network. The Journal of

Supercomputing, pp.1-22.

[5] Kumar, R.D., Julie, E.G., Robinson, Y.H., Vimal, S. and Seo, S., 2021. Recognition

of food type and calorie estimation using neural network. The Journal of

Supercomputing, pp.1-22.



