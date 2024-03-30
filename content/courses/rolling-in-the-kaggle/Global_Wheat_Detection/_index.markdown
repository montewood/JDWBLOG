---
date: "2024-03-30"
linkTitle: "Global Wheat Detection"
title: "Global Wheat Detection"
type: book 
weight: 20
---

<center> 

**Link** : <https://www.kaggle.com/competitions/global-wheat-detection/data>

</center>

# Dataset Description

&nbsp;More details on the data acquisition and processes are available at https://arxiv.org/abs/2005.02162

## What should I expect the data format to be?

&nbsp;The data is images of wheat fields, with bounding boxes for each identified wheat head. Not all images include wheat heads / bounding boxes. The images were recorded in many locations around the world.

&nbsp;The CSV data is simple - the image ID matches up with the filename of a given image, and the width and height of the image are included, along with a bounding box (see below). There is a row in train.csv for each bounding box. Not all images have bounding boxes.

&nbsp;Most of the test set images are hidden. A small subset of test images has been included for your use in writing code.

## What am I predicting?
&nbsp; You are attempting to predict bounding boxes around each wheat head in images that have them. If there are no wheat heads, you must predict no bounding boxes.

## Files
 - **train.csv** - the training data
 - **sample_submission.csv** - a sample submission file in the correct format
 - **train.zip** - training images
 - **test.zip** - test images

## Columns

 - **image_id** - the unique image ID
 - **width**, **height** - the width and height of the images
 - **bbox** - a bounding box, formatted as a Python-style list of [xmin, ymin, width, height]   
etc.


