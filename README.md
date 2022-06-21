# Object_Detection

**Primary Objective and Team Considerations**

The primary objective going into this project was to build an object detection algorithm that self-driving cars can use to safely navigate the roads with optimal performance.

Our team initially considered approaching this problem from a CNN standpoint since we were aware of how it involved multiclass classification. At that point, we weren't too familiar with region-based CNNs or open-source neural networks like DarkNet, so we wanted to try building and training a simple model and see what percentage of images it was able to classify correctly.

We primarily used 2 datasets, the first one being CIFAR10 and the second one being ImageNet. We initially used CIFAR10 to build a simple image classifier that could predict the probability of an input image being a car or a truck. As far as using ImageNet, we utilized a pre-trained CNN known as VGG16 that has been trained on millions of images from the ImageNet database and used it with a custom neural network that we built from scratch. We were overall able to achieve an accuracy of about 94.3 percent on the testing dataset.

**Algorithm Employment**

Since this problem involved object detection for autonomous vehicles, we wanted to build and train a model that could help us achieve as high of an accuracy as possible. As a result, we utilized multiple algorithms along the way as a means of gauging which one delivered the highest accuracy. Overall there were 4 different ones that we used and those were:

1. 2-layer Convolutional Neural Network
![](2-Layer%20CNN.png)
We first built a simple 2-layer neural network and trained the model on a cars and trucks dataset for about 20-30 epochs. To check the accuracy of the model, we normalized the data to speed up convergence before fitting it to the network and also converted all class labels into a one-hot encoded vector. Our model achieved an accuracy of almost 96.2% on the training dataset, but only 74.3% on the testing dataset. We observed that the model was overfitting quite a bit, so we wanted to try a different algorithm to check if we could increase the accuracy of the image classifier.
3. Sliding Window Algorithm
4. Transfer Learning model
5. YOLO(DarkNet)

