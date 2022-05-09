# PixelClassification
## File Structure
```
|-- ROOT
  |-- README.md
  |-- PixelClassifier_Report.pdf
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



# Report
## Objective
Train a probabilistic color model from pixel data to distinguish among red, green, and blue pixels.
### Detailed Tasks

<img width="821" alt="Screen Shot 2022-05-09 at 16 21 50" src="https://user-images.githubusercontent.com/92130976/167491156-ccf3a540-1b30-4d42-aa95-644c7e332f78.png">

## Problem Formulation
Consider we have an pixel X with 3 layers of colors: (R, G, B), and we need to identify which color it belongs to. Let the true image colors to be an 3 ∗ 1 one-hot encoder Y . We want to train a 3 ∗ 3 weight w which project X into Yˆ depending on XT ∗ w, which is the probabilistic prediction. The goal is finding w s.t. min|Yˆ − Y |.

For color classification, we want to generate a 3 ∗ 3 weight w to minimize the loss. Hence, we are using the following algorithm for training.

## Algorithm
Since we only generate the probability of our model output, we use argmax to get the position of probability. For example, if the probability we generate is [0.8, 0.1, 0.1], the agrmax output would be [1, 0, 0].

<img width="271" alt="Screen Shot 2022-05-09 at 16 30 11" src="https://user-images.githubusercontent.com/92130976/167492595-f09bc376-00a8-499e-bc40-a3b9352143d3.png">

## Results
For Pixel Classification, the train pre- cision is 0.9926908500270709; validation result is 0.975904; and test result is 9.879518072289157/10.




