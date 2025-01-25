# ml-zoomcamp-capstone-2-2024
## Description
This is DataTalksClub [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) capstone 2 project repo.

It is trained on human faces dataset and tries to pedict if person in the image url is young or old. 

Dataset credit: https://www.kaggle.com/datasets/goubeast/age-prediction-av

This model has different potential use cases:

Ex: Preferential access to senior citizens based on face recognition etc.

Here are step by step details about how we build optimal model to predict young or old probability. 

## 1. Data preparation, cleanup
Data downloaded from above kaggle source was in slightly different format: [raw_data](./raw_data)

It had images and corresponding csv mapping image to class OLD YOUNG or MIDDLE age.

This script turns it into format compatible for our model training: [segregate_images.py](./segregate_images.py)

We also use this script to split data in train and test folders in 75 % and 25 % ratio: [splitdata.py](./splitdata.py)

Here is the final tree strucure of data: [data](./data)

Note: we have ommited middle age images data for simplicity purpose.
```bash
data
├── test
│   ├── old
│   └── young
└── train
    ├── old
    └── young
```    
## 2. Training a model
Code here: [notebook.ipynb](./notebook.ipynb)

Note: You will need GPU support like saturn cloud to run this notebook.
https://app.community.saturnenterprise.io/

On CPU it will be very slow and prone to crash.

## 3. Saving model
We saved models with checkpoint.

We use model with best accuracy checkpoint: [xception_v1_08_0.893.keras](./xception_v1_08_0.893.keras)

Convert it to lighter TF-Lite model: [model.tflite](./model.tflite)

Here is a jupyter notebook which uses model.tflite and does prediction.

[tf_lite.ipynb](./tf_lite.ipynb)

And converted python script:

[tf_lite.py](./tf_lite.py)
