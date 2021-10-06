---
layout: page
title: MultiON Challenge at CVPR 2021
description: Our solution ranked first at the Multi-Object Navigation Challenge for Embodied AI Workshop at CVPR 2021!
img: /assets/img/multion_chall_2.png
importance: 1
category: Participant
---

**Date:** February, 2021

## Multi-Object Navigation
The Multi-Object Navigation (MultiON) task was introduced in [this paper](https://arxiv.org/abs/2012.03912). The goal is for an agent initialized in an unknown photorealistic scene to reach a sequence of target external objects randomly placed within the environment. 

**Why is MultiON interesting ?**
* A sequential task is interesting as it requires the agent to remember and to map potential objects it might have seen while exploring the environment, as reasoning on them might be required in a later stage.
* Using external objects as goals prevents the agent from leveraging knowledge about the environment layouts, thus focusing solely on memory.

## MultiON Challenge
The [MultiON Challenge](http://multion-challenge.cs.sfu.ca/) was organized as part of the [Embodied AI Workshop](https://embodied-ai.org/) at CVPR 2021. The Leaderboard can be found on the Challenge Website.

For more details, watch the [video](https://www.youtube.com/watch?v=ghX5UDWD1HU) presenting the Competition and the results during the workshop.

**Our solution**

Our solution (see [paper](https://arxiv.org/abs/2107.06011)) won the first place. We introduce auxiliary tasks guiding the learning of spatial reasoning abilities in agents trained with reinforcement learning. More specifically, we show that 
<!-- mapping capacities can emerge by  --> training agents to predict the distance to, direction towards target objects that were previously seen within the episode, as well as estimating if the current goal to find has already been within the agent's field of view so far improves significantly the overall performance. Our submission to the Challenge was an agent equipped with projective geometry to map its environment, and trained with our auxiliary objectives.

The video presenting our solution can be found [here](https://www.youtube.com/watch?v=boDaAORoKho).

<!-- <div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/fig_method_gt_dist.pdf' | relative_url }}" alt="" title="example image"/>
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div> -->
