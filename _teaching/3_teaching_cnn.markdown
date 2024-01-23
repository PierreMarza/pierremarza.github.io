---
layout: distill
title: Introduction to Deep Learning
description: Convolutional Neural Networks
date: 2024-01-22

authors:
  - name: Pierre Marza
    url: "https://pierremarza.github.io/"
    affiliations:
      name: INSA, Lyon

bibliography: 2018-12-22-distill.bib

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

# 1. Lecture
The lecture slides are available [here](https://pierremarza.github.io/teaching/lyon1_m1_deep_learning_cnn.pdf).

# 2. Practical
## Context
The goals of this session are to practice with **implementing a Convolutional Neural Network**, **understanding the involved computations**, and more generally, to **build a full Deep Learning pipeline in PyTorch** to train a model on a given dataset.

## Installation
We will be coding with **Python3** and will use the **Pytorch** library.

To install Pytorch on your local machine, follow this [link](https://pytorch.org/get-started/locally/)

## Convolutional Neural Networks
Convolutional Neural networks (CNNs) are powerful neural models that take advantage of the convolution operator to learn to extract information from raw images. You can refer to your [lecture](https://pierremarza.github.io/teaching/lyon1_m1_deep_learning_cnn.pdf) for more information about CNNs, as well as many great online resources.

## Inductive Biases in Neural Networks
Why would we want to use a CNN when dealing with images? We could also use a simpler Multi-Layer Perceptron (MLP), where each neuron in a layer aggregates information from all neurons in the previous layer. An important notion in Machine Learning is known as [**inductive bias**](https://en.wikipedia.org/wiki/Inductive_bias) or **model prior**. This corresponds to the prior knowledge you, as a designer, incorporate in the model you are building.

Take some time to think about what is the prior knowledge incorporated into a CNN, that is not in an MLP.

{::options parse_block_html="true" /}
<details><summary markdown="span">**An answer**</summary>
The assumption that data has a spatial underlying structure, known as **Spatial Inductive Bias** is used in CNNs. Indeed, the convolution operation aggregates information from only the local spatial neighborood around the center of the filter. Models equipped with such inductive bias are particularly well suited to extract information from the pixels of an image.
</details>
<br/>
{::options parse_block_html="false" /} 

## The CIFAR-10 dataset
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a well-known dataset of RGB images. It is composed of **60000 32x32 colour images** labelled as belonging to one of **10 classes**. You can find **6000 images per class**. There are **50000 training images** and **10000 test images**. If you are looking for a dataset with more classes, you can look at [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) that, as indicated by its name, contains 100 semantic classes.

The following Pytorch code takes advantage of [**torchvision**](https://pytorch.org/vision/stable/index.html). It allows you to create 3 datasets (train, val, test) and to apply a normalization to the images. The names of the 10 classes in the CIFAR-10 dataset are also given for you to use later.

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split train set into a train and validation sets
train_size = int(0.75*len(train_set))
valid_size = len(train_set) - train_size
train_set, valid_set = torch.utils.data.random_split(train_set, [train_size, valid_size])

# Ground-Truth classes in the CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

## Data loading and visualisation
It is important to visualise the data you will be working on. Moreover, when training and evaluating your model, you will need to load data from your training, validation and test sets respectively. To do so, we will use the [**DataLoader**](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) class from [**torch.utils.data**](https://pytorch.org/docs/stable/data.html).

Start by **implementing 3 dataloaders** for your training, validation and test sets.

Finally, write a function that takes a batch of image tensors as inputs and display them, along with their associated labels. You can use [**matplotlib.pyplot**](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html) to do so.

## Designing a Convolutional Neural Network
It is now time to build a CNN! Write a class inheriting from [**torch.nn.Module**](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). Be careful of the dimensions of input tensors and the dimensions of your desired output.
Then, you can play with different hyperparameters, such as the number of layers, and hyperparameters of the [**torch.nn.Conv2d**](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) layer (number of output channels, kernel size, stride, padding, etc.)

## Dimension of feature maps in CNN
It is important to understand the convolution operations going on inside your neural network.

### Formula
Let's denote the size of input and output tensors of a convolution layer along axis $x$ as $I_x$ and $O_x$, and the respective kernel size, padding and stride as $K_x$, $P_x$ and $S_x$. **Write the formula that gives you $O_x$ as a function of $I_x$, $K_x$, $P_x$ and $S_x$.**

**IMPORTANT: Before moving on to the next step, call me to present and eventually discuss this result!**

### Code
Now, you can write a function that takes as input a tensor shape, as well as the hyperparameters of the layer and outputs the size of the output tensor. You can then check your function is correct by comparing its returned value with the real shapes of tensors within the forward pass of your neural network.

## Loss function and optimizer
The next step is to define a [**loss function**](https://pytorch.org/docs/stable/nn.html#loss-functions) that is suited to the problem you want to solve, in our case **multi-class classification**. Then you have to choose an [**optimizer**](https://pytorch.org/docs/stable/optim.html). You are encouraged to try different ones to compare them. You can also study the impact of different hyperparameters of the optimizer (learning rate, momentum, etc.)

## Training loop
It is now to time to write the code for **training and validating your model**. You must iterate through your training data using your dataloader, and compute forward and backward passes on given data batches.
Don't forget to log your training as well as validation losses (the latter is mainly used to tune hyperparameters).

## Visualizing your training with Tensorboard
A useful tool to visualize your training is [**Tensorboard**](https://www.tensorflow.org/tensorboard/). You can also have a look at solutions such as [**Weights & Biases**](https://wandb.ai/site), but we will focus on the simpler Tensorboard for now.
You can easily use Tensorboard with Pytorch by looking at [**torch.utils.tensorboard**](https://pytorch.org/docs/stable/tensorboard.html)

## Saving and loading a Pytorch model
Once training is completed, it can be useful to save the weights of your neural network to use it later. The following [tutorial](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) explains how you can do this. Now, try to save and then load your trained model.

## Testing your model
You must now **evaluate the performance of your trained model** on the **test set**. To this end, you have to iterate through test samples, and perform forward passes on given data batches. You might want to compute the **test loss**, but also any **accuracy-related metrics** you are interested in. You could also **visualize some test samples** along with the **output distribution of your model**.