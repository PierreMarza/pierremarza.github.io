---
layout: page
title: Sim2Real Domain Transfer
description: We studied Sim2Real Domain Transfer in the context of Visual Navigation.
img: /assets/img/psat_img.png
importance: 1
category: Academic research projects
---

**Date:** 2019--2020

**Advisor:** [Christian Wolf](https://perso.liris.cnrs.fr/christian.wolf/), INSA Lyon, France

Training agents that perform Visual Navigation tasks can be done with Deep Reinforcement Learning. In such case, leveraging simulated environments has several advantages, such as training speed or safety. However, when deployed in the real world, these agents can suffer from the well-known Sim2Real gap.

In this Master's research project, we studied this Sim2Real gap, by collecting a dataset of aligned RGB-D pairs between the camera of a real robot and a scanned version of the environment it involves in (leveraging the Habit Simulator from FAIR). We also experimented with different methods to try to close this gap.

**Abstract:**
The robot navigation task in a known or unknown envi-ronment is a trending subject in the Computer Vision field.Various  studies  showed  promising  results  with  Reinforce-ment Learning, often coupled with Deep Learning methods.However, these approaches usually require extensive train-ing to be done with the robot in the anticipated environment.The field experimented with Simulated Environment Learn-ing as it allows for very fast bootstrapping and learning.This specific task requires then to use the learned model ina real environment.  An efficient way to close the gap be-tween simulated and real data distributions is through Do-main Transfer, i.e.  methods trying to build models that arerobust to a change of characteristics inside the considereddata.  In this paper, we provide a comparison between sev-eral state-of-the-art approaches to reduce Domain Shift inthe  case  of  our  Robot  Navigation  task.   We  focus  on  ad-versarial methods, leveraging the generative capabilities ofthe Variational AutoEncoder and CycleGAN architecturesto learn general latent representations.  Unsupervised Do-main Adaptation for robot pose estimation is also experi-mented. The experiments are to be done on CITI’s providedrobot  equipped  with  a  Microsoft  Kinect  video  and  depthsensor.   We  thus  introduce  CITI-Sim2Real,  a  new  datasetadapted to our problem.

A report summarizing our work can be found [here](/al-folio/assets/pdf/P_SAT.pdf).