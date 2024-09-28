# nextgen
Objective

This project aims to recognize handwritten digits in real-time using a deep learning model. Code is made of two parts
        1.Training the model using the MNIST dataset to classify digits from 0 to 9.
       2.Implementing a real-time interface using Pygame and OpenCV,  users can draw digits, which are then recognized by the trained model.
       
Project Overview

Part 1: Training the Model

•	Dataset: MNIST, a popular dataset of handwritten digits.
•	Model Architecture:
o	The model uses convolutional neural networks (CNN) with layers such as Conv2D, MaxPool2D, Flatten, Dropout, and Dense.
o	Model  is optimized using functions ,callbacks like EarlyStopping and ModelCheckpoint.
•	Objective: Train the model which can recognize digits (0-9) and save and choose the best-performing model out of them

Part 2: Real-Time Digit Recognition

•	Interface: Pygame is used to create a drawing panel where users can draw digits with their mouse.
•	Prediction: OpenCV captures the drawing, preprocesses it (resizing and padding), and the dl model predicts the digit.
•	Display: Predicted digit will be displayed along with the number identified on top of it

Technologies Used
•	Python
•	Keras / TensorFlow: For building and training the dl model.
•	MNIST Dataset: Used for training and testing the model.
•	Pygame: For creating a real-time drawing interface.
•	OpenCV: For processing the drawn input and converting it to the required format for the model to predict.
