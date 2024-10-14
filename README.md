# Deep Learning Lab Practice IV - Assignments

**Course:** Final Year Information Technology (2019 Course)  
**University:** Savitribai Phule Pune University, Pune  
**Subject Code:** 414447 - Lab Practice IV  
**Teaching Scheme:**
- Practical (PR): 02 hrs/week
- Credits: 01 credit
- PR: 25 Marks, TW: 25 Marks  

**Subject Teacher:** Prof. Neeta Shirsat

---

## Course Objectives:
The course is designed to:
1. Formulate deep learning problems corresponding to different applications.
2. Apply deep learning algorithms to solve problems of moderate complexity.
3. Apply algorithms to real-world problems, optimize models, and report expected accuracy.

## Course Outcomes:
By completing these assignments, students will be able to:
- CO1: Learn and use various Deep Learning tools and packages.
- CO2: Build and train deep Neural Network models.
- CO3: Apply CNN, RNN, and Autoencoders to real-world problems.
- CO4: Evaluate the performance of deep learning models.

---

## Assignments

### Assignment 1: Study of Deep Learning Packages
**File:** `Assignment_01.ipynb`

#### Description:
This assignment focuses on documenting the distinct features and functionality of the following Deep Learning packages:
- TensorFlow
- Keras
- Theano
- PyTorch

#### Steps Covered:
1. **Package Documentation:** Analyze and document key differences.
2. **Dataset Usage:** Use a sample dataset to demonstrate implementation in these frameworks.

#### Dataset:
- MNIST / CIFAR10

---

### Assignment 2: Implementing Feedforward Neural Networks with Keras and TensorFlow
**File:** `Assignment_02.ipynb`

#### Description:
This assignment involves building a simple Feedforward Neural Network using Keras and TensorFlow.

#### Steps Covered:
1. **Import Packages:** Keras, TensorFlow, and others.
2. **Load Data:** Load MNIST or CIFAR10 dataset.
3. **Define Network Architecture:** Using Keras to define the layers.
4. **Train Model:** Use Stochastic Gradient Descent (SGD).
5. **Evaluate:** Measure performance and accuracy.
6. **Plot Results:** Visualize training loss and accuracy.

#### Dataset:
- MNIST / CIFAR10

---

### Assignment 3: Building an Image Classification Model
**File:** `Assignment_03.ipynb`

#### Description:
In this assignment, we build an image classification model through four stages.

#### Steps Covered:
1. **Loading Data:** Preprocess and load the image dataset.
2. **Defining Model Architecture:** Create the neural network.
3. **Training the Model:** Using training data.
4. **Evaluating Performance:** Measure accuracy and performance.

#### Dataset:
- Custom Image Dataset (or MNIST/CIFAR10)

---

### Assignment 4: Anomaly Detection Using Autoencoders
**File:** `Assignment_04.ipynb`

#### Description:
Use an Autoencoder neural network to implement anomaly detection.

#### Steps Covered:
1. **Import Libraries:** Keras, TensorFlow.
2. **Upload Dataset:** Load the data for anomaly detection.
3. **Encoder:** Converts input into a latent representation.
4. **Decoder:** Converts the latent representation back to the original input.
5. **Compile Model:** Set the optimizer, loss function, and metrics.
6. **Evaluate Model:** Check performance on detecting anomalies.

#### Dataset:
- Custom Anomaly Detection Dataset

---

### Assignment 5: Continuous Bag of Words (CBOW) Model Implementation
**File:** `Assignment_05.ipynb`

#### Description:
Implement the Continuous Bag of Words (CBOW) model for natural language processing tasks.

#### Steps Covered:
1. **Data Preparation:** Preprocess the text data.
2. **Generate Training Data:** Create the training data for CBOW.
3. **Train Model:** Train the CBOW model.
4. **Output:** Analyze the output vectors.

#### Dataset:
- Custom Text Corpus

---

### Assignment 6: Object Detection Using Transfer Learning of CNN Architectures
**File:** `Assignment_06.ipynb`

#### Description:
Implement object detection using transfer learning with pre-trained CNN models.

#### Steps Covered:
1. **Load Pre-trained CNN:** Use a pre-trained model (e.g., VGG16).
2. **Freeze Parameters:** Freeze the lower convolutional layers.
3. **Add Custom Classifier:** Add new trainable layers for the specific task.
4. **Train Model:** Fine-tune the classifier.
5. **Tune Hyperparameters:** Adjust parameters and unfreeze more layers if needed.

#### Dataset:
- Custom Object Detection Dataset

---

## Setup Instructions:
To run the assignments, follow these steps:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/sagargiradkar/dl

