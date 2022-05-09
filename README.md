# PixelClassification
## File Structure
```
|-- ROOT
  |-- README.md
  |-- Pipfile
  |-- requirements.txt
  |-- run_tests.py
  |-- tests
  |   |-- test_simple.py
  |   |-- testset
  |-- pixel_classification
  |   |-- data.zip
  |   |-- generate_rgb_data.py
  |   |-- pixel_classifier.py
  |   |-- pixel_weight.npy
  |   |-- requirements.txt
  |   |-- test_pixel_classifier.py
```

## Objective
Train a probabilistic color model from pixel data to distinguish among red, green, and blue pixels.

## Detailed Tasks

<img width="821" alt="Screen Shot 2022-05-09 at 16 21 50" src="https://user-images.githubusercontent.com/92130976/167491156-ccf3a540-1b30-4d42-aa95-644c7e332f78.png">

# Report
## Overview:
With the increasing trend of building an environment-friendly community, col- lecting recyclables has become very impor- tant. However, finding all blue recycling bins over one road when collecting is still a problem for the human employee. People might not identify each container precisely over a long time. Hence, it will be better if we have an excellent classifier to help hu- man drivers detect the recycle bins and im- prove the efficiency of collecting recyclables.
In this report, I proposed one model that can easily and quickly find and make bounding blue recycling bins. This model first uses one color classifier based on lo- gistic regression, morphological operations, and bounding area extension to identify blue recycling bins.

<img width="406" alt="Screen Shot 2022-05-09 at 16 25 05" src="https://user-images.githubusercontent.com/92130976/167491646-7b3772bc-6979-468a-b34d-b01086603d34.png">

Figure 1: Overview of Training and Testing Process

## Algorithm

<img width="271" alt="Screen Shot 2022-05-09 at 16 30 11" src="https://user-images.githubusercontent.com/92130976/167492595-f09bc376-00a8-499e-bc40-a3b9352143d3.png">






