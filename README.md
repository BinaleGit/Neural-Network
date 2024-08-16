# TensorFlow Fashion and Number Prediction Projects

This repository contains two separate projects for predicting fashion items and handwritten digits using TensorFlow. The projects utilize transfer learning and custom neural networks to classify the Fashion MNIST and MNIST datasets, respectively.

## Project Structure

- **Tensor_clothes_Predict/**: This directory contains the code and model for predicting fashion items from the Fashion MNIST dataset.
  - `app.py`: The main script for training and predicting fashion items.
  - `fashion_model_parameters.pkl`: The saved model parameters after training.

- **Tensor_Number_Predict/**: This directory is intended for the number prediction project, similar to the fashion prediction, but working with the MNIST dataset.

- **requirements.txt**: Contains the Python dependencies required to run both projects.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.7 or higher
- TensorFlow
- Matplotlib
- Numpy

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt

```
## Running the Fashion MNIST Prediction (Tensor_clothes_Predict)
1. Training the Model
If you need to train the model from scratch, you can run the app.py script in the Tensor_clothes_Predict directory. This script will train a neural network using transfer learning with the pre-trained MobileNetV2 model on the Fashion MNIST dataset.

```bash
cd Tensor_clothes_Predict
python app.py
```
2. Using the Pre-Trained Model
After training, the model parameters are saved in the fashion_model_parameters.pkl file. If you want to load the pre-trained model and make predictions without re-training, you can run the app.py script directly:
```bash
python app.py
```
This will load the saved model and make predictions on the test dataset.

## Running the MNIST Number Prediction (Tensor_Number_Predict)
This directory is intended for the MNIST number prediction project. The structure and approach are similar to the fashion prediction project, and it should be set up in a similar manner.

## Repository Overview
 - `app.py`: This is the main script for both the fashion and number prediction projects. It contains the code for loading the dataset, preprocessing the data, training the model, and making predictions.
  - `fashion_model_parameters.pkl`: This file stores the trained model parameters for the Fashion MNIST dataset. It is automatically created after the model is trained.
 - `requirements.txt`: Lists all the necessary Python libraries and dependencies required to run the projects.


