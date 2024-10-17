---
layout: distill
title: "AutoNeRF: Training Implicit Scene Representations with Autonomous Agents"
description: IROS 2024
date: 2023-10-01
category: Research projects
img: /assets/img/autonerf/teaser_figure.png

authors:
  - name: Pierre Marza
    url: "https://pierremarza.github.io/"
    affiliations:
      name: INSA Lyon
  - name: Laëtitia Matignon
    url: "https://perso.liris.cnrs.fr/laetitia.matignon/"
    affiliations:
      name: UCBL
  - name: Olivier Simonin
    url: "http://perso.citi-lab.fr/osimonin/"
    affiliations:
      name: INSA Lyon
  - name: Dhruv Batra
    url: "https://faculty.cc.gatech.edu/~dbatra/"
    affiliations:
      name: Georgia Tech, Meta AI
  - name: Christian Wolf
    url: "https://chriswolfvision.github.io/www/"
    affiliations:
      name: Naver Labs Europe
  - name: Devendra Singh Chaplot
    url: "https://devendrachaplot.github.io/"
    affiliations:
      name: Meta AI

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

[**Paper**](https://arxiv.org/abs/2304.11241){: .btn}
[**Code**](https://github.com/PierreMarza/autonerf){: .btn}
[**NeRF4ADR Workshop (ICCV 2023) poster**](/assets/img/autonerf/nerf4adr_poster.pdf){: .btn}

[**IROS 2024 video**](https://youtu.be/CJz2_pAeSKk){: .btn}

## Abstract
Implicit representations such as Neural Radiance Fields (NeRF) have been shown to be very effective at novel view synthesis. However, these models typically require manual and careful human data collection for training. In this paper, we present AutoNeRF, a method to collect data required to train NeRFs using autonomous embodied agents. Our method allows an agent to explore an unseen environment efficiently and use the experience to build an implicit map representation autonomously. We compare the impact of different exploration strategies including handcrafted frontier-based exploration and modular approaches composed of trained high-level planners and classical low-level path followers. We train these models with different reward functions tailored to this problem and evaluate the quality of the learned representations on four different downstream tasks: classical viewpoint rendering, map reconstruction, planning, and pose refinement. Empirical results show that NeRFs can be trained on actively collected data using just a single episode of experience in an unseen environment, and can be used for several downstream robotic tasks, and that modular trained exploration models significantly outperform the classical baselines.

<img src="/assets/img/autonerf/teaser_video.gif" width="100%" />

<iframe width="100%" height="395" src="https://www.youtube.com/embed/CJz2_pAeSKk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
</iframe>

## Reconstructing house-scale scenes
We start by illustrating the possibility of autonomously reconstructing complex large-scale environments such as apartments or houses from the continuous representations trained on data collected by agents exploring a scene using a modular policy. You can visualize RGB and semantics meshes (extracted from NeRF models) of 5 scenes from the Gibson val set. The semantics head of the NeRF models was trained with GT labels from the Habitat simulator.

<!-- Import the component -->
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.0.1/model-viewer.min.js"></script>

<!-- Code adapted from https://modelviewer.dev/examples/augmentedreality/index.html#customButton -->
<model-viewer
  alt="Collierville"
  src="/assets/img/autonerf/Corozal_rgb.glb"
  style="width: 100%; height: 600px; background-color: #404040"
  poster="/assets/img/autonerf/Corozal_rgb.png"
	exposure=".8"
  auto-rotate
  camera-controls
>
  <div class="slider">
    <div class="slides">
      <button class="slide selected" onclick="switchSrc(this, 'Corozal_rgb')" style="background-image: url('/assets/img/autonerf/Corozal_rgb.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Corozal_sem')" style="background-image: url('/assets/img/autonerf/Corozal_sem.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Collierville_rgb')" style="background-image: url('/assets/img/autonerf/Collierville_rgb.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Collierville_sem')" style="background-image: url('/assets/img/autonerf/Collierville_sem.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Darden_rgb')" style="background-image: url('/assets/img/autonerf/Darden_rgb.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Darden_sem')" style="background-image: url('/assets/img/autonerf/Darden_sem.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Markleeville_rgb')" style="background-image: url('/assets/img/autonerf/Markleeville_rgb.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Markleeville_sem')" style="background-image: url('/assets/img/autonerf/Markleeville_sem.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Wiconisco_rgb')" style="background-image: url('/assets/img/autonerf/Wiconisco_rgb.png');">
      </button>
      <button class="slide" onclick="switchSrc(this, 'Wiconisco_sem')" style="background-image: url('/assets/img/autonerf/Wiconisco_sem.png');">
      </button>
    </div>
  </div>
</model-viewer>

<script type="module">
  const modelViewer = document.querySelector("model-viewer");

  window.switchSrc = (element, name) => {
    const base = "/assets/img/autonerf/" + name;
    modelViewer.alt = name;
    modelViewer.src = base + '.glb';
    modelViewer.poster = base + '.png';
    const slides = document.querySelectorAll(".slide");
    slides.forEach((element) => {element.classList.remove("selected");});
    element.classList.add("selected");
  };
</script>

<style>
  /* This keeps child nodes hidden while the element loads */
  :not(:defined) > * {
    display: none;
  }

  .slider {
    width: 100%;
    text-align: center;
    overflow: hidden;
    position: absolute;
    bottom: 16px;
  }

  .slides {
    display: flex;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
  }

  .slide {
    scroll-snap-align: start;
    flex-shrink: 0;
    width: 100px;
    height: 100px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    background-color: #fff;
    margin-right: 10px;
    border-radius: 10px;
    border: none;
    display: flex;
  }

  .slide.selected {
    border: 2px solid #4285f4;
  }

  .slide:focus {
    outline: none;
  }

  .slide:focus-visible {
    outline: 1px solid #4285f4;
  }
</style>


## Exploration Policy
The trained policy aims to allow an agent to explore a 3D scene to collect a sequence of 2D RGB and semantic frames and camera poses, that will be used to train the continuous scene representation. Following previous work, we adapt a modular policy composed of a *Mapping* process that builds a *Semantic Map*, a *Global Policy* that outputs a global waypoint from the semantic map as input, and finally, a *Local Policy* that navigates towards the global goal.

<p align="center">
    <img src="/assets/img/autonerf/policy_figure.png" width="80%" />
</p>

## Reward definitions
We consider different reward signals for training the Global Policy tailored to our task of scene reconstruction, and which differ in the importance they give to different aspects of the scene. All these signals are computed in a self-supervised fashion using the metric map representations built by the exploration policy:

* **Explored area** --- *Ours (cov.)* optimizes the coverage of the scene, i.e. the size of the explored area.
* **Obstacle coverage** --- *Ours (obs.)* optimizes the coverage of obstacles in the scene. It targets tasks where obstacles are considered more important than navigable floor space, which is arguably the case when viewing is less important than navigating.
* **Semantic object coverage** --- *Ours (sem.)* optimizes the coverage of the semantic classes detected and segmented in the semantic metric map. This reward removes obstacles that are not explicitly identified as a notable semantic class.
* **Viewpoints coverage** --- *Ours (view.)* optimizes for the usage of the trained implicit representation as a dense and continuous representation of the scene usable to render arbitrary new viewpoints, either for later visualization as its own downstream task, or for training new agents in simulation. To this end, we propose to maximize coverage not only in terms of agent positions, but also in terms of agent viewpoints. Such reward functions does not only encourage to cover all objects in the scene, but also to view them from different viewpoints.

## Downstream tasks
Prior work on implicit representations generally focused on two different settings: (i) evaluating the quality of a neural field based on its new view rendering abilities given a dataset of (carefully selected) training views, and (ii) evaluating the quality of a scene representation in robotics conditioned on given (constant) trajectories, evaluated as reconstruction accuracy. We cast this task in a more holistic way and more aligned with our scene understanding objective. We evaluate the impact of trajectory generation (through exploration policies) directly on the quality of the representation, which we evaluate in a goal-oriented way through multiple tasks related to robotics.

We present the different downstream tasks and qualitative results for *Ours (obs.)*. A quantititative comparison between policies is presented in the paper.

### Task 1: Rendering
This task is the closest to the evaluation methodology prevalent in the neural field literature. We evaluate the rendering of RGB frames and semantic segmentation. Unlike the common method of evaluating an implicit representation on a subset of frames within the trajectory, we evaluate it on a set of uniformly sampled camera poses within the scene, independently of the trajectory taken by the policy. This allows us to evaluate the representation of the complete scene and not just its interpolation ability.

Below are rendering examples, where the semantic head of the NeRF model was trained with GT semantic labels from the Habitat simulator. The same NeRF model will be used to provide qualitative examples in the next subsections.
<p align="center">
    <img src="/assets/img/autonerf/viz_rendering_figure_supp_mat.png" width="100%" />
</p>


### Task 2: Metric Map Estimation
While rendering quality is linked to perception of the scene, it is not necessarily a good indicator of its structural content, which is crucial for robotic downstream tasks. We evaluate the quality of the estimated structure by translating the continuous representation into a format, which is very widely used in map-and-plan baselines for navigation, a binary top-down (bird's-eye-view=BEV) map storing occupancy and semantic category information and compare it with the ground-truth from the simulator. We evaluate obstacle and semantic maps using accuracy, precision, and recall.
<p align="center">
    <img src="/assets/img/autonerf/viz_map_estimation.png" width="90%" />
</p>

### Task 3: Planning
Using maps for navigation, it is difficult to pinpoint the exact precision required for successful planning, as certain artifacts and noises might not have a strong impact on reconstruction metrics, but could lead to navigation problems, and vice-versa. We perform goal-oriented evaluation and measure to what extent path planning can be done on the obtained top-down maps.
<p align="center">
    <img src="/assets/img/autonerf/viz_planning.png" width="80%" />
</p>

### Task 4: Pose Refinement
This task involves correcting an initial noisy camera position and associated rendered view and optimizing the position until a given ground-truth position is reached, which is given through its associated rendered view only. The optimization process therefore leads to a trajectory in camera pose space. This task is closely linked to visual servoing with a *eye-in-hand* configuration, a standard problem in robotics, in particular in its *direct* variant, where the optimization is directly performed over losses on the observed pixel space.
<p align="center">
    <img src="/assets/img/autonerf/camera_pose_refinement_video.gif" width="100%" />
</p>
