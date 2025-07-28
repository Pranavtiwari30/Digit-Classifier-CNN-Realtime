🔢 Digit Classifier: CNN-Based Real-Time Handwritten Digit Recognition
🧠 Objective
This project aims to recognize handwritten digits in real-time using a Convolutional Neural Network (CNN). The system consists of two main components:

Model Training – Train a deep learning model using the MNIST dataset to classify digits (0–9).

Real-Time Inference Interface – Implement a live digit recognition interface using Pygame and OpenCV, where users can draw digits with a mouse.

🚀 Project Overview
📌 Part 1: Model Training
Dataset: MNIST — a widely used dataset of handwritten digits.

Model: A CNN architecture built using Keras/TensorFlow with layers like:

Conv2D, MaxPooling2D, Dropout, Flatten, and Dense

Optimization:

Uses EarlyStopping and ModelCheckpoint callbacks for better training performance.

Goal: Accurately classify digits from 0 to 9 and save the best model for deployment.

🖥️ Part 2: Real-Time Digit Recognition
Interface: Built using Pygame for drawing digits in real-time.

Preprocessing: Uses OpenCV to resize and format the drawn image to match the model’s input shape.

Prediction: The trained model classifies the digit and displays it on the screen in real-time.

🛠️ Tech Stack
Python

Keras / TensorFlow – Deep Learning Framework

OpenCV – Image preprocessing

Pygame – Drawing interface for digit input

MNIST Dataset – For training the digit classifier

📸 Demo Preview
(Add screenshots or a GIF of real-time digit recognition in action here if you have one)

📂 Folder Structure
bash
Copy
Edit
├── model_training/
│   └── train_model.py
├── realtime_inference/
│   └── app.py
├── saved_model/
│   └── digit_cnn_model.h5
├── README.md
└── requirements.txt
🚀 How to Run
Install requirements:

bash
Copy
Edit
pip install -r requirements.txt
Train the model (optional if already trained):

bash
Copy
Edit
python model_training/train_model.py
Run real-time digit recognition:

bash
Copy
Edit
python realtime_inference/app.py
👨‍💻 Author
Pranav Tiwari
GitHub: @Pranavtiwari30
