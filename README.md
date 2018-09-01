# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./exploration-visualization.png "Visualize"
[image2]: ./bar-chart.png "BarChart"
[image3]: ./preprocessed.png "preprocessing"
[image4]: ./1-ahead-only.jpg "1-ahead-only"
[image5]: ./2-no-passing.jpg "2-no-passing"
[image6]: ./3-no-entry.jpg "3-no-entry"
[image7]: ./german-traffic-sign-caution-roadworks-71151565.jpg "german-traffic-sign-caution-roadworks-71151565"
[image8]: ./img6.jpg "img6"
[image9]: ./new1.jpg "new1"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project](https://github.com/anandagrawal2909/TrafficSignClassifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 x 32 x 3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

At first, I plotted some 20 random images from the training set and here it is:

![alt text][image1]

Then, I plotted a bar chart to visualize the distribution of the images with respect to their classes. Here is the chart:
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

*Available data is NOT preprocssed, so have applied Grayscaing (converting to single color map) to reduce size.
*Also applied normalisation to so that variation between data is reducced.

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer1         		|     Convolution1								| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
|						|												|
 
| Layer2         		|     Convolution2	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 14x14x6 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
|						|												|


| Layer3         		|     Fully-connected1	        				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 5x5x16 = 400   								| 
| output	      	    | 120						 					|
|						|												|

| Layer4         		|     Fully-connected1	        				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 120   										| 
| output	      	    | 84						 					|
|						|												|

| Layer4         		|     Fully-connected1	        				| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 84   											| 
| output	      	    | 43						 					|
|						|												|





#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

*Type of Optimizer = AdamOptimizer
** Reason ::Since its most popular and easy to use.
*Batch Size = 128
** Reason :: This had based results during training
*Number of Epochs = 50
** Reason :: Accuracy was stable by 50 Epochs
*Learning Rate = 0.001
** Reason :: parameter was found in hyperparameters tunning.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.5 
* test set accuracy of 93.9

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Speed limit 70    	| Speed limit 70 										|
| Straight or left 		| Straight or left											|
| Road Work ahead		| Road Work ahead			 						|
| 80 km/hr 		 		| 80 km/hr      							|
| No Passing      		| No Passing    									| 
|:---------------------:|:---------------------------------------------:| 


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 
