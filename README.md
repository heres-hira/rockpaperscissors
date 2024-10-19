# Rock Paper Scissors - Machine Learning Project

This project applies deep learning to classify hand gestures (rock, paper, and scissors) using image data. The model uses computer vision techniques to recognizer each gesture and predict the outcome of a rock-paper-scissors game.

## Features
- Classifies images of hand gestures (rock, paper, scissors) using a Convolutional Neural Network (CNN)
- Trained on a dataset of hand gesture images
- Uses Keras and TensorFlow libraries for model building and training
- Can be extended to include more complex features like live prediction with a camera

## Process Overview
1. **Data Preparation**: The dataset is organized into three categories (rock, paper, and scissors), and split into training and validation sets. The images are pre-processed by rescaling pixel values to a range between 0 and 1, and data augmentation techniques like zoom, rotation, and flipping are applied to increase dataset diversity and help the model generalize better.
2. **Model Architecture**: A Convolutional Neural Network (CNN) is built using Keras. The architecture consists of several convolutional layers to extract features from the images, followed by pooling layers to reduce the dimensionality, and finally dense (fully connected) layers to perform classification. The output layer uses a softmax activation function to classify the image as either rock, paper, or scissors.
3. **Training and Evaluation**: The model is trained on the training set, where it learns to recognize patterns in the hand gestures. During training, the model adjusts its weights through backpropagation to minimize the loss function, which in this case could be categorical cross-entropy. The model’s performance is evaluated on the validation set during each epoch to monitor overfitting or underfitting.
4. **Prediction**: Once trained, the model can predict the class of a new hand gesture image (rock, paper, or scissors). When given an image, the model processes it through the CNN layers, extracting features and making a final prediction based on the highest output probability.

## Dependencies
- TensorFlow
- Keras
- Numpy
- Matplotlib
