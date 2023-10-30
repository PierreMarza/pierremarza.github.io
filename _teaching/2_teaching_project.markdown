---
layout: distill
title: Deep Learning Project
description: Comparing CNNs, RNNs and Transformers on sequential data
date: 2023-10-11

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

## Timeline
* **October, 16**: Form groups.
* **Before October, 23**: One person per group sends me an [**e-mail**](mailto:pierre.marza@insa-lyon.fr) with:
  * The names of all group members.
  * The compute resources you plan to use.
* **November, 3**: **Written report** presenting a related work (previous work in the literature, see below), the performed data pre-processing, the models you considered (architectures, hyperparameters, etc.), quantitative results, drawn conlusions and explanations.
* **November, 6**: **Oral presentation** (10 min + 5 min for questions).

## Compute
You can choose among 2 options:
* Use your own laptop, particularly if you have a GPU.
* [Google Colab](https://colab.research.google.com/?utm_source=scs-index) is a way to access a GPU for free.

## Context
Temporal data is everywhere. Being able to leverage sequential data is thus an important challenge. In this course, we saw three types of neural networks: CNNs, RNNs and Transformers. The goal of this project is to present a rigorous quantitative comparison of the performance of each of the three approaches to handle temporal data.

The broad question you will have to answer in this project is: **How do the three types of architectures compare on a given sequential dataset?** You will select different architectures and hyperparameters for the three classes of models in order to provide a rigorous and fair comparison. Such scientific comparative study is very valuable when starting a project dealing with a new dataset. Comparing different approaches fairly and rigorously will help you pick the best one.

## Dataset
In this project, you will have to compare CNN, RNN and Transformer models on the task of **binary movie review sentiment classification**, i.e. given a textual review of a movie, predicting whether the given review is positive or negative. You will use the [**Large Movie Review Dataset**](http://ai.stanford.edu/~amaas/data/sentiment/). From the previous link, you can download the dataset and access a *README* to learn about the structure of the dataset folder.

* The dataset contains **unsupervised samples**. **You should not use them**, but **only supervised train and test sets**. And also, **remember what we saw in the course about the importance of building and using a proper validation set!**

* The dataset also provides **review scores**. You **don't have to use those**, only treat this as a **binary classification problem** (classification with 2 classes: *positive* and *negative*).

* **You can't use any pre-trained model**. Everything should be **trained from scratch!**

## Project milestones
### Related Work
At the beginning of the project, you should spend some time on the **literature**. **Each member in the group should select and read at least one research paper** that is relevant and presents a method that could be applied to your problem. **Such related work should be presented in your written report**.

### Text to vectors
In the project, you will need to **convert textual reviews to vectors**. You are asked to **compare different approaches** to do so. This comparison will be presented in the final report, along with the details of your data pre-processing pipeline.

### CNN vs RNN vs Transformer
As already mentionned, the core question you will try to answer is the following: **What works best between CNNs, RNNs and Transformers on this particular dataset?** You should not compare only one model of each, but rather pick different variants and hyperparameters to provide a fair and rigorous comparison.
