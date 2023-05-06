# Real-Time-Mask-Detection
This project focuses on creating an object detection model in real time for detecting whether or not someone is wearing a facemask.

The [dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) used to train this model consists of 12.000 images for 2 class, where one class consists of people wearing masks and the other class are maskless people.

The model uses a Sequential TensorFlow model with the aid of transfer learning from ResNet 152V2.

Currently, the model managed to achieve a 0.99% accuracy on the train and test set, but still has issues in detecting someone from far away or if the image is significantly distorted.
