### Brain Tumor Classification Using CNN

#### Overview
The **Brain Tumor Classification CNN** project focuses on building a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor. By utilizing deep learning, the model aims to provide an automated way to assist in the early detection of brain tumors from medical images, ultimately helping doctors with diagnoses.

---

<img src="https://github.com/user-attachments/assets/68dfe9eb-19f1-4f64-993b-146664eb9547" width="300" />

<img src="https://github.com/user-attachments/assets/db4616a5-8ffc-4107-8376-b2a8fbfcf924" width="300" />




#### Project Flow

1. **Data Loading:**
   - The dataset consists of brain tumor MRI images organized into four main classes: Glioma, Meningioma, Pituitary, and No Tumor.
   - Images are loaded using TensorFlow's `ImageDataGenerator`, which also applies image augmentation techniques like horizontal flipping, zooming, and shearing to avoid overfitting.

2. **Data Preprocessing:**
   - Image data is scaled from pixel values of 0–255 to a normalized range of 0–1 for better model convergence.
   - Duplicate images are removed to ensure data integrity and prevent training biases.
   - The classes are encoded numerically, and the distribution of the data across each class is visualized using a bar chart.

3. **Model Architecture:**
   - The model is built using TensorFlow and consists of a convolutional encoder (for feature extraction) followed by fully connected layers that classify the images into one of the four tumor categories.
   - The model uses **categorical cross-entropy** as the loss function and **softmax activation** in the output layer for multi-class classification.

4. **Model Training:**
   - The CNN model is trained for 10 epochs using the training set while validating on the test set. We leverage batch processing to handle large datasets efficiently.
   - Model performance is evaluated using accuracy, precision, recall, and AUC metrics.

5. **Evaluation:**
   - The model’s performance is evaluated on the test set. Metrics like **accuracy**, **precision**, **recall**, and **AUC** are printed after each evaluation.

6. **Prediction:**
   - The model is used to predict the class of unseen MRI images, providing valuable insights for tumor detection in medical diagnostics.

---

#### Key Features & Results

- **Data Augmentation**: Applied random transformations such as shear, zoom, and horizontal flip to increase model robustness.
- **Model Performance**:
   - **Test Accuracy**: ~[accuracy score]
   - **Test Precision**: ~[precision score]
   - **Test Recall**: ~[recall score]
   - **Test AUC**: ~[AUC score]

- **Class Distribution Visualization**:
   - A bar graph is generated to show the distribution of the classes in the training set, ensuring balanced class representation for the model.

<img src="https://github.com/user-attachments/assets/430c054f-c782-400b-aa1d-c2fefc4884f7" width="450" />

---

#### Libraries & Tools Used

- **TensorFlow & Keras**: For building and training the CNN model.
- **Matplotlib**: For visualizing class distributions and other data insights.
- **Scikit-learn**: For evaluation metrics like precision, recall, and F1-score.
- **ImageDataGenerator**: For image augmentation and scaling.

---

#### Future Improvements

- **Hyperparameter Tuning**: Implementing techniques such as grid search to optimize model parameters.
- **Transfer Learning**: Leveraging pre-trained models (e.g., VGG16 or ResNet) to enhance model performance on smaller datasets.
- **Deployment**: Packaging the model into a web application or a standalone tool for easy access by healthcare professionals.

---

This project aims to combine computer vision with healthcare, providing a tool for doctors and clinicians to enhance decision-making in brain tumor detection.
