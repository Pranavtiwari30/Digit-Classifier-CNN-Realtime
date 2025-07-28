# Digit-Classifier-CNN-Realtime

## Objective

This project aims to recognize handwritten digits in real-time using a deep learning model. Code is made of two parts:  
  1. Training the model using the MNIST dataset to classify digits from 0 to 9.  
  2. Implementing a real-time interface using Pygame and OpenCV, where users can draw digits, which are then recognized by the trained model.

---

## Project Overview

### Part 1: Training the Model

• Dataset: MNIST, a popular dataset of handwritten digits.  
• Model Architecture:  
 o The model uses convolutional neural networks (CNN) with layers such as Conv2D, MaxPool2D, Flatten, Dropout, and Dense.  
 o The model is optimized using callbacks like EarlyStopping and ModelCheckpoint.  
• Objective: Train a model capable of recognizing digits (0–9) and save the best-performing model for real-time use.

---

### Part 2: Real-Time Digit Recognition

• Interface: Pygame is used to create a drawing panel where users can draw digits using their mouse.  
• Prediction: OpenCV captures the drawing, preprocesses it (resizing and padding), and the deep learning model predicts the digit.  
• Display: The predicted digit is shown on the screen along with the drawn input.

---

## Technologies Used

• Python  
• Keras / TensorFlow – For building and training the deep learning model  
• MNIST Dataset – Used for training and testing  
• Pygame – For creating a real-time drawing interface  
• OpenCV – For processing the drawn input and converting it to model-ready format  

---

## Folder Structure

```bash
├── model_training/
│   └── train_model.py               # Script for training the CNN on MNIST
├── realtime_inference/
│   └── app.py                       # Real-time interface using Pygame + OpenCV
├── saved_model/
│   └── digit_cnn_model.h5          # Trained CNN model
├── README.md
└── requirements.txt
