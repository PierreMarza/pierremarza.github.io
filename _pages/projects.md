---
layout: page
permalink: /projects/
title: Projects
description: Some projects I worked on.
nav: true
display_categories: [Academic research projects]
horizontal: false
---
<div class="projects">
  {% if site.enable_projects_categories and page.display_categories %}
  <!-- Display categorized projects -->
    {% for category in page.display_categories %}
      <h2 class="category">{{category}}</h2>
      {% assign categorized_projects = site.projects | where: "category", category %}
      {% assign sorted_projects = categorized_projects | sort: "importance" %}
      <!-- Generate cards for each projects -->
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-2">
          {% for projects in sorted_projects %}
            {% include projects_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for projects in sorted_projects %}
            {% include projects.html %}
          {% endfor %}
        </div>
      {% endif %}
    {% endfor %}

  {% endif %}

</div>