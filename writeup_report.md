
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "Center Lane Driving"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image6]: ./examples/1img.jpg "Normal Image"
[image7]: ./examples/1flipped.jpg "Flipped Image"
[image8]: ./examples/loss.png "Loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-24) 
The model includes RELU layers to introduce nonlinearity (code line 83), and the data is normalized in the model using a Keras lambda layer (code line 81). 


####2. Attempts to reduce overfitting in the model

The final model does not contain dropout layers, it was discarded after being tested and result in worse results than with the current model (model.py lines 21).
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving in opposite direction of the track as a way to generalize and train the model for recovering scenarios.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was based on the model suggested by Udacity.
I tested more complex models, based on NVIDIA and other models, but I did not have better results or even had difficulty training due to lack of memory on my system.
The beginning of the model crops the image to remove its sides and focus on the important information and avoid trees and the sky.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
The final step was to run the simulator to see how well the car was driving around track one. There were very few spots where the vehicle start to fell off the track but was able to recovery. To improve the driving behavior in these cases, recovery situations were used as training data.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 80-92) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture:

|Layer (type)                     | Output Shape         | Param #    | Connected to                    |
|            :---:                |     :---:            |  :---:     |           :---:                 |
|lambda_1 (Lambda)                |(None, 160, 320, 3)   |0           |lambda_input_1[0][0]             |
|cropping2d_1 (Cropping2D)        |(None, 65, 320, 3)    |0           |lambda_1[0][0]                   |
|convolution2d_1 (Convolution2D)  |(None, 31, 158, 24)   |1824        |cropping2d_1[0][0]               |
|convolution2d_2 (Convolution2D)  |(None, 14, 77, 36)    |21636       |convolution2d_1[0][0]            |
|convolution2d_3 (Convolution2D)  |(None, 5, 37, 48)     |43248       |convolution2d_2[0][0]            |
|convolution2d_4 (Convolution2D)  |(None, 3, 35, 64)     |27712       |convolution2d_3[0][0]            |
|convolution2d_5 (Convolution2D)  |(None, 1, 33, 64)     |36928       |convolution2d_4[0][0]            |
|flatten_1 (Flatten)              |(None, 2112)          |0           |convolution2d_5[0][0]            |
|dense_1 (Dense)                  |(None, 100)           |211300      |flatten_1[0][0]                  |
|dense_2 (Dense)                  |(None, 50)            |5050        |dense_1[0][0]                    |
|dense_3 (Dense)                  |(None, 10)            |510         |dense_2[0][0]                    |
|dense_4 (Dense)                  |(None, 1)             |11          |dense_3[0][0]                    |

Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to deal with situations like these.
These images show what a recovery looks like (left lane recovery):

![alt text][image3]
![alt text][image4]

To augment the data sat, I flipped images and angles thinking that this would improve de model and helps to generalize.
For example, the original image and flipped one:

![alt text][image6]
![alt text][image7]

After the collection process, I had 13963 images file. Network used 41889 processeded images to train.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss graph. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image8]