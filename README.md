# Brain-Tumor-Image-Classification-Using-CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: glioma, meningioma, pituitary, and no tumor. The goal is to aid in the detection of different types of brain tumors, providing a tool for medical professionals. T

<img src="https://github.com/user-attachments/assets/68dfe9eb-19f1-4f64-993b-146664eb9547" width="300" />

<img src="https://github.com/user-attachments/assets/db4616a5-8ffc-4107-8376-b2a8fbfcf924" width="300" />


## Project Flow

1. **Data Loading**:
   
   -The dataset consists of brain tumor MRI images organized into four main classes: Glioma, Meningioma, Pituitary, and No Tumor.
   
   -Images are loaded using TensorFlow's ImageDataGenerator, which also applies image augmentation techniques like horizontal flipping, zooming, and shearing to avoid overfitting.



3. **Data Preprocessing**:
   -Image data is scaled from pixel values of 0–255 to a normalized range of 0–1 for better model convergencee.
   -Duplicate images are removed to ensure data integrity and prevent training biases.
   -The classes are encoded numerically, and the distribution of the data across each class is visualized using a bar chart.

   
4. **Model Architecture:**:
   The model is built using TensorFlow and consists of a convolutional encoder (for feature extraction) followed by fully connected layers that classify the images into one of the four tumor categories.
The model uses categorical cross-entropy as the loss function and softmax activation in the output layer for multi-class classification.  
   <img width="997" alt="age_graph" src="https://github.com/user-attachments/assets/680d1914-9ef1-4c6c-b275-44bb1df19514">

6. **Normalization**:
   - We normalize numerical features (`age`, `hypertension`, `heart_disease`, `avg_glucose_level`, and `bmi`) using `StandardScaler`, ensuring that all features are on the same scale.

7. **Addressing Class Imbalance**:
   - To tackle class imbalance, we use SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of the minority class.

8. **Train-Test Split**:
   - The dataset is split into training and testing sets, which is super important for evaluating our models.

9. **Model Training**:
   - We train two machine learning models:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**

10. **Model Evaluation**:
   - We calculate and display metrics like accuracy, precision, recall, and F2-score for both models.
   - A bar graph compares the accuracy of the models.  
   <img width="695" alt="Screenshot 2024-09-28 at 9 03 28 PM" src="https://github.com/user-attachments/assets/0c4a9a1c-2ad6-467a-a986-f7232552d79a">


11. **User Input for Prediction**:
   - There’s a function that asks users to input their health and lifestyle details to predict their likelihood of having a stroke.
   - We normalize the input data using the fitted scaler to keep everything consistent.

11. **Prediction**:
    - Finally, we use the trained models to predict the likelihood of a stroke based on user input and show the results.

## Running the Project

Wanna give it a shot? Here’s how:

1. Make sure you have the required libraries installed. You can get them with pip:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
