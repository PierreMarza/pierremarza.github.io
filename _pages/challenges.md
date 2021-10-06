---
layout: page
title: Challenges
permalink: /challenges/
description: Competitions I have been involved in.
nav: true
display_categories: [Participant]
horizontal: false
---
<div class="challenges">
  {% if site.enable_challenge_categories and page.display_categories %}
  <!-- Display categorized challenges -->
    {% for category in page.display_categories %}
      <h2 class="category">{{category}}</h2>
      {% assign categorized_challenges = site.challenges | where: "category", category %}
      {% assign sorted_challenges = categorized_challenges | sort: "importance" %}
      <!-- Generate cards for each challenge -->
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-2">
          {% for challenge in sorted_challenges %}
            {% include challenges_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for challenge in sorted_challenges %}
            {% include challenges.html %}
          {% endfor %}
        </div>
      {% endif %}
    {% endfor %}

  {% else %}
  <!-- Display challenges without categories -->
    {% assign sorted_challenges = site.challenges | sort: "importance" %}
    <!-- Generate cards for each challenge -->
    {% if page.horizontal %}
      <div class="container">
        <div class="row row-cols-2">
        {% for challenge in sorted_challenges %}
          {% include challenges_horizontal.html %}
        {% endfor %}
        </div>
      </div>
    {% else %}
      <div class="grid">
        {% for challenge in sorted_challenges %}
          {% include challenges.html %}
        {% endfor %}
      </div>
    {% endif %}

  {% endif %}

</div>
