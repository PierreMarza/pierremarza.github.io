---
layout: distill
title: Deep Learning Project
description: Semi-supervised Image Classification
date: 2021-01-02

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

## Context
Annotating data can be costly, and large corpuses of carefully labeled data are not always available. An important domain of study in Deep Learning is thus to learn as much as posible from unlabelled data. This is exactly what you are going to do in this project.

## Dataset
We will use the Stanford [**STL-10 dataset**](https://cs.stanford.edu/~acoates/stl10/). It is composed of **100k unlabeled images** along with **500 labeled training images per class**. This is the only data you can use for training your model.

In the context of this project, **you cannot use models that were pre-trained on other datasets**.

## Timeline
* **January, 3**: Form **4 groups**.
* **Before January, 5**: One person per group sends me an e-mail with:
  * The names of all group members.
  * The list of selected research papers along with a few keywords describing their content (related to the project).
  * The compute resources you plan to use.
* **January, 25**: Oral presentation (20 min + 10 min for questions).

## Related Work
At the beginning of the project, you should spend some time on the **litterature**. **Each member in the group should select and read at least one research paper** that is relevant and presents a method that could be applied to your problem.

## Baselines
After becoming more familiar with the literature, an interesting first step is often to **implement a baseline approach**, i.e. a method that is as simple as possible and that solves the task reasonably well. This will be your comparison point for new ideas you might want to try.

## Compute
You can choose among 3 options:
* Use your own laptop, particularly if you have a GPU.
* [Google Colab](https://colab.research.google.com/?utm_source=scs-index) is a way to access a GPU for free.
* Each group has an account on our cluster, so you might want to run jobs there (you will be limited to a single GPU per group).