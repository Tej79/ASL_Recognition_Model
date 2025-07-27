# ASL Alphabet Recognition with CNN

This project is a computer vision application that recognizes American Sign Language (ASL) letters using a custom Convolutional Neural Network (CNN). It involves training a model from scratch on an image dataset of hand gestures representing the ASL alphabet (A–Z, excluding J and Z as they involve movement), and evaluating its performance on unseen data.

## 🧠 Overview

The primary goal of this project is to build a deep learning model that can accurately classify static hand signs corresponding to the ASL alphabet. This is accomplished through:

- **Data preprocessing and visualization**
- **Model development with CNN architecture**
- **Training with validation tracking**
- **Data Augmentation to boost performance**
- **Performance evaluation on test data**

## 📊 Dataset

The dataset used for training and testing are csv files containing the pixel values that the image consists of, each representing a single letter from the ASL alphabet (excluding dynamic gestures like "J" and "Z" as they involve movement). The dataset is divided into training and testing files. The final test data consists of raw .png images of the hand signs for a and b, which passed to the model and the model processes the data to extract pixel values on its own.

## 🛠️ Technologies Used

- Python 3
- NumPy & Pandas
- Matplotlib
- TensorFlow

## 🧪 Model Architecture

The custom CNN model includes:

- **Convolutional layers** with ReLU activation
- **Max-pooling layers** for spatial reduction
- **Dropout layers** to prevent overfitting
- **Fully connected layers** for classification

## 📈 Results

- The model achieved high accuracy (>98%) on the test set after training on augmented data.
- Training and validation curves indicate strong generalization without significant overfitting.

## 🚀 How to Run
The best way to run these files and any files with heavy models is using Google Colab since it's a virtual environment and more importantly, you can utilise high performance GPUs and TPUs to speed up processing times. I'd highly discourage running these files and models locally as they are very demanding and would likely take hours to process a single image. (Trust me, I've tried)

That said, you can clone and open the .ipynb files from this repo directly from colab and run the files there once connected to a virtual runtime. You'd also need to download the data folder into your runtime so the notebook can access them. 

If you only wish to view the process and the results achieved, you can simply preview the .ipynb file in GitHub itself.


## Ackowledgments
The dataset used in this project is available from the website [Kaggle](http://www.kaggle.com).
