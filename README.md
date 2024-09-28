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
Setup and Installation
1.	Clone the repository
git clone https://github.com/your-username/real-time-ocr-digits.git
cd real-time-ocr-digits
2.	Install the required dependencies:
pip install -r requirements.txt
Part 1: Training the Model
Code Breakdown
The training code trains on the MNIST dataset. It includes functions for plotting the dataset, labelling the data set, building on CNN, training it with callbacks, and evaluating its performance using performance metrics
•	Model:
o	Conv2D and MaxPooling layers -for feature extraction.
o	Dropout layer -to avoid overfitting.
o	Softmax activation function- for multi-class classification.
From code:
model = Sequential([
    Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
    MaxPool2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dropout(0.25),
    Dense(10, activation="softmax"))] 
•	Callbacks:
o	EarlyStopping to stop training if validation accuracy stops improving.
o	ModelCheckpoint to save the best model based on validation accuracy.
From code:
es = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
mc = ModelCheckpoint('bestmodel.h5', monitor='val_accuracy', save_best_only=True)
Running the Training
Model can be run using:
python train_model.py
it will train the model on the MNIST dataset and save the best model as  (bestmodel.h5)
Part 2: Real-Time Digit Recognition
Code Breakdown
The real-time interface is built using Pygame. It allows users to draw digits, captures the drawn image, and passes it through the trained model for prediction. The recognized digit is then displayed on the screen.
•	Pygame for Interface: Allows users to draw digits on a panel.
•	OpenCV for Preprocessing: Captures the drawing, resizes and normalizes it before feeding it into the model.
From displaysurrfare:
img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
image = cv2.resize(img_arr, (28, 28)) / 255.0
label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
Running the Digit Recognition Interface
When model is being trained we can start by enter command:
python digit_recognition.py
It will launch a pannel where you can draw digits using mouse, and model will predict digit
Example:
Draw any number using mouse on the pannel  
The model will analyse the drawing and display the predicted digit on the screen.
