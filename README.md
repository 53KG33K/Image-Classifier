# Image Classifier Training and Prediction

This repository contains a program to train an image classifier using a pre-trained neural network model and a separate program to make predictions using the trained model.

## Overview

The repository includes two main functionalities:
1. **Training**: Train an image classifier on a dataset of images.
2. **Prediction**: Use the trained model to predict the class of an input image.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- Required Python modules (imported in the scripts)

## Training the Model

To train the model, execute the following command in your terminal:

```bash
python train.py data_dir --save_dir save_directory --arch model_architecture --learning_rate lr --hidden_units hidden_units --epochs num_epochs --gpu

```

To predict the model, execute the following command in your terminal:

```bash
python predict.py input_image checkpoint --top_k K --category_names mapping_file --gpu

python predict.py flowers/test/1/image_06752.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

