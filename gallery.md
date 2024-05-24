---
layout: default
title: 图片展览
permalink: /gallery/
---

# 图片展览

<div class="gallery">
  {% for item in site.gallery %}
    <div class="gallery-item">
      <a href="{{ item.image }}">
        <img src="{{ item.image }}" alt="{{ item.title }}" />
      </a>
      <div class="caption">
        <h3>{{ item.title }}</h3>
        <p>{{ item.description }}</p>
      </div>
    </div>
  {% endfor %}
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/slick-carousel/1.8.1/slick.css" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/slick-carousel/1.8.1/slick.min.js"></script>
<script>
$(document).ready(function(){
  $('.gallery').slick({
    infinite: true,
    slidesToShow: 3,
    slidesToScroll: 3,
  });
});
</script>

<style>
.gallery {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-around;
}

.gallery-item {
  margin: 5px;
}

.gallery-item img {
  max-width: 100%;
  height: auto;
  display: block;
}

.caption {
  text-align: center;
  margin-top: 10px;
}
</style>
