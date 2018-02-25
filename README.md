# MNIST_Data_Using_CNN_and_Tuning

This is the R code for tuning CNN using MINST Fashion data set. The CNN code contributed includes the following contents:

Upload data;
CNN Function;
Tuning;
Running Function
Many thanks!

From other sources:

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Why we made Fashion-MNIST The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

To Serious Machine Learning Researchers Seriously, we are talking about replacing MNIST. Here are some good reasons:

MNIST is too easy. Convolutional nets can achieve 99.7% on MNIST. Classic machine learning algorithms can also achieve 97% easily. Check out our side-by-side benchmark for Fashion-MNIST vs. MNIST, and read "Most pairs of MNIST digits can be distinguished pretty well by just one pixel." MNIST is overused. In this April 2017 Twitter thread, Google Brain research scientist and deep learning expert Ian Goodfellow calls for people to move away from MNIST. MNIST can not represent modern CV tasks, as noted in this April 2017 Twitter thread, deep learning expert/Keras author François Chollet.

In the R script, you will see the following:

R INTERFACE TO KERAS BEGIN: 

Keras is a high-level neural networks API developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research. Keras has the following key features:

MNIST Example:

The MNIST dataset is included with Keras and can be accessed using the dataset_mnist() function. Here we load the dataset then create variables for our test and training data:

DEFINE THE MODEL:

The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers. We begin by creating a sequential model and then adding layers using the pipe (%>%) operator

TRAINING AND EVALUATION:

Keras provides a vocabulary for building deep learning models that is simple, elegant, and intuitive. Building a question answering system, an image classification model, a neural Turing machine, or any other model is just as straightforward.

TUNING: 

Tuning is a process of finding the optimal parameters for the machine. Here I present code by myself to find the number of hidden nodes for the first three hidden layers. The net is coded using 4 layers with the last layer to have 10 hidden units, a present from LeCun's work. 

Reference: Coding of Keras interace refers to RStudio Keras website page.
All rights are reserved for Yiqiao Yin.
