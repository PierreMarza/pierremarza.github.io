---
layout: page
permalink: /teaching/
title: Teaching
description: Materials for taught courses.
nav: true
# years: [2021]
display_categories: [2021]
horizontal: false
---
<!-- <div class="publications">
{% assign courses = site.teaching%}
{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
    {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-2">
          {% for course in courses %}
            {% include teaching_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for course in courses %}
            {% include teaching.html %}
          {% endfor %}
        </div>
      {% endif %}
{% endfor %}

</div> -->

<div class="teaching">
  {% if site.enable_teaching_categories and page.display_categories %}
  <!-- Display categorized teaching -->
    {% for category in page.display_categories %}
      <h2 class="category">{{category}}</h2>
      {% assign categorized_teaching = site.teaching | where: "category", category %}
      {% assign sorted_teaching = categorized_teaching | sort: "importance" %}
      <!-- Generate cards for each teaching -->
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-2">
          {% for teaching in sorted_teaching %}
            {% include teaching_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for teaching in sorted_teaching %}
            {% include teaching.html %}
          {% endfor %}
        </div>
      {% endif %}
    {% endfor %}

  {% endif %}

</div>