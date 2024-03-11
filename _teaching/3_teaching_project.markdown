---
layout: distill
title: Artificial Intelligence & Data Analysis
description: Deep Learning project
date: 2024-03-07

authors:
  - name: Pierre Marza
    url: "https://pierremarza.github.io/"
    affiliations:
      name: INSA, Lyon
  - name: Johan Peralez

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

## Timeline
* **Before March, 18**: Make groups of 2 people and fill the following [Google sheet](https://docs.google.com/spreadsheets/d/1g6aqVYruR6g-hutjSjE_z7C2618t1Qhm2WaS9o0zgOE/edit?usp=sharing):
  * Pick a group id (you can choose the one you want, not important).
  * Specify group member names.
  * Rank each of the 8 projects from most wanted (1) to least wanted (8).
* **April, 9**: **Oral presentation** (10 min + 5 min for questions).

## Compute
You can choose among 2 options:
* Use your own laptop, particularly if you have a GPU.
* [Google Colab](https://colab.research.google.com/?utm_source=scs-index) is a way to access a GPU for free.

## List of subjects
Below is the list of subjects you can choose from. Please note that **two groups can't work on the same subject!**

### 1. Handwritten digits classification (Computer Vision, Supervised learning)
<img src="/assets/img/lyon1_dl_project/mnist.png" width="100%" />
#### Data
You can use the code below to create train and test Pytorch datasets. It is then your job to explore the provided data. Also, don't forget to create a proper validation set from a part of your training data.
```python
import torchvision
train_dataset = torchvision.datasets.MNIST(
  "data", 
  train=True, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)

test_dataset = torchvision.datasets.MNIST(
  "data", 
  train=False, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)
```

#### Project milestones
You will have to properly do the following:
1. Building a training/validation/test data split. Do not change the provided test set. You should keep the official test set of the task!
2. Validating your model and training hyperparameters on the validation set.
3. Testing your best model (based on the validation performance) on the test set. For this, you'll go beyond simple loss and accuracy metrics, but will also consider other metrics.
4. Visualizing the outputs of your model, and understanding its errors.

### 2. Handwritten letters classification (Computer Vision, Supervised learning)
<img src="/assets/img/lyon1_dl_project/emnist.png" width="100%" />
#### Data
You can use the code below to create train and test Pytorch datasets. It is then your job to explore the provided data. Also, don't forget to create a proper validation set from a part of your training data.
```python
import torchvision
train_dataset = torchvision.datasets.EMNIST(
  "data", 
  split="letters",
  train=True, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)

test_dataset = torchvision.datasets.EMNIST(
  "data",
  split="letters",
  train=False, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)
```

#### Project milestones
You will have to properly do the following:
1. Building a training/validation/test data split. Do not change the provided test set. You should keep the official test set of the task!
2. Validating your model and training hyperparameters on the validation set.
3. Testing your best model (based on the validation performance) on the test set. For this, you'll go beyond simple loss and accuracy metrics, but will also consider other metrics.
4. Visualizing the outputs of your model, and understanding its errors.

### 3. Japanese symbols classification (Computer Vision, Supervised learning)
<img src="/assets/img/lyon1_dl_project/kmnist.png" width="100%" />
#### Data
You can use the code below to create train and test Pytorch datasets. It is then your job to explore the provided data. Also, don't forget to create a proper validation set from a part of your training data.
```python
import torchvision
train_dataset = torchvision.datasets.KMNIST(
  "data", 
  train=True, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)

test_dataset = torchvision.datasets.KMNIST(
  "data", 
  train=False, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)
```

#### Project milestones
You will have to properly do the following:
1. Building a training/validation/test data split. Do not change the provided test set. You should keep the official test set of the task!
2. Validating your model and training hyperparameters on the validation set.
3. Testing your best model (based on the validation performance) on the test set. For this, you'll go beyond simple loss and accuracy metrics, but will also consider other metrics.
4. Visualizing the outputs of your model, and understanding its errors.

### 4. Clothing articles classification (Computer Vision, Supervised learning)
<img src="/assets/img/lyon1_dl_project/fashion_mnist.png" width="100%" />
#### Data
You can use the code below to create train and test Pytorch datasets. It is then your job to explore the provided data. Also, don't forget to create a proper validation set from a part of your training data.
```python
import torchvision
train_dataset = torchvision.datasets.FashionMNIST(
  "data", 
  train=True, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)

test_dataset = torchvision.datasets.FashionMNIST(
  "data", 
  train=False, 
  download=True,
  transform=torchvision.transforms.ToTensor(),
)
```

#### Project milestones
You will have to properly do the following:
1. Building a training/validation/test data split. Do not change the provided test set. You should keep the official test set of the task!
2. Validating your model and training hyperparameters on the validation set.
3. Testing your best model (based on the validation performance) on the test set. For this, you'll go beyond simple loss and accuracy metrics, but will also consider other metrics.
4. Visualizing the outputs of your model, and understanding its errors.


### 5. Mountain Car (Reinforcement Learning) 
<img src="/assets/img/lyon1_dl_project/mountain_car.png" width="35%" />

Environnement : [https://gymnasium.farama.org/environments/classic_control/mountain_car](https://gymnasium.farama.org/environments/classic_control/mountain_car)

Difficulté : Les mesures sont continues (position et vitesse). Surtout, la fonction coût / récompense, est nulle pour la plupart des paires état-action (sparse reward).

Etape 1 : Discrétiser les mesures pour appliquer l’algorithme de  Q learning tabulaire vu en cours.


### 6. Cart Pole (Reinforcement Learning)
<img src="/assets/img/lyon1_dl_project/cartpole.png" width="35%" />

Environnement : https://gymnasium.farama.org/environments/classic_control/cart_pole/

Difficulté : Les mesures sont continues (position et vitesse chariot, position et vitesse angulaire de la tige).
Surtout les mesures sont nombreuses et définies sur un ensemble non borné.

Etape 1 : Appliquer l’algorithme de Deep-Q learning vu en cours.


### 7. Pendulum (Reinforcement Learning)
<img src="/assets/img/lyon1_dl_project/pendulum.png" width="25%" />

Environnement : https://gymnasium.farama.org/environments/classic_control/pendulum/

Difficulté : Les actions et les mesures sont continues (position et vitesse).

Etape 1 : Proposer et implémenter un contrôleur de l’automatique classique (étudié dans une autre U.E.).

### 8. Acrobot (Reinforcement Learning)
<img src="/assets/img/lyon1_dl_project/acrobot.png" width="25%" />

Environnement : https://gymnasium.farama.org/environments/classic_control/acrobot/

Difficulté : Les mesures sont continues et nombreuses.
La fonction coût / récompense, est nulle pour la plupart des paires état-action (sparse reward).

Etape 1 : Appliquer l’algorithme de Deep-Q learning vu en cours.
