Here are answers and explanations for your assignment on implementing Feedforward Neural Networks. I'll cover the key points for both practical implementation and viva-style questions.

---

### **Practical-Oriented Steps and Answers**

1. **Import Necessary Libraries**
   ```python
   import numpy as np
   import pandas as pd
   from sklearn.preprocessing import LabelBinarizer
   from sklearn.metrics import classification_report
   from tensorflow.keras import models, layers, optimizers, datasets, backend as K
   ```

2. **Load and Preprocess the Data (MNIST)**
   ```python
   (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
   
   # Flatten the images and normalize
   train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype("float32") / 255
   test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype("float32") / 255
   
   # One-hot encode the labels
   lb = LabelBinarizer()
   train_labels = lb.fit_transform(train_labels)
   test_labels = lb.transform(test_labels)
   ```

3. **Define the Network Architecture Using Keras**
   ```python
   model = models.Sequential([
       layers.Dense(128, activation='sigmoid', input_shape=(784,)),
       layers.Dense(64, activation='sigmoid'),
       layers.Dense(10, activation='softmax')
   ])
   ```

4. **Compile and Train the Model Using SGD**
   ```python
   model.compile(optimizer=optimizers.SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
   history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
   ```

5. **Evaluate the Network**
   ```python
   test_loss, test_accuracy = model.evaluate(test_images, test_labels)
   print("Test accuracy:", test_accuracy)
   ```

6. **Plot Training Loss and Accuracy**
   ```python
   import matplotlib.pyplot as plt
   
   # Plot loss
   plt.plot(history.history['loss'], label='Training Loss')
   plt.plot(history.history['val_loss'], label='Validation Loss')
   plt.legend()
   plt.title('Training and Validation Loss')
   plt.show()

   # Plot accuracy
   plt.plot(history.history['accuracy'], label='Training Accuracy')
   plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
   plt.legend()
   plt.title('Training and Validation Accuracy')
   plt.show()
   ```

7. **Calculate Precision, Recall, F1-score, and Support**
   ```python
   predictions = model.predict(test_images)
   predictions = np.argmax(predictions, axis=1)
   test_labels = np.argmax(test_labels, axis=1)
   
   print(classification_report(test_labels, predictions))
   ```

---

### **Viva Question Answers**

1. **What is a Feedforward Neural Network?**
   - A Feedforward Neural Network (FNN) is a type of artificial neural network where connections between nodes do not form cycles. It’s the simplest type of neural network architecture, where data moves in one direction—from input to output.

2. **How Does a Feedforward Neural Network Work?**
   - In a Feedforward Neural Network, data passes through layers of nodes with each layer connected to the next. The input layer receives the input data, hidden layers transform the data, and the output layer produces the final prediction.

3. **Real-Time Scenarios Where FNNs Are Used**
   - Handwritten digit recognition (e.g., MNIST dataset).
   - Fraud detection in banking transactions.
   - Facial recognition systems.

4. **Components of a Feedforward Neural Network**
   - **Input Layer**: Receives data.
   - **Hidden Layers**: Perform computations and feature extraction.
   - **Output Layer**: Generates the final output.
   - **Weights**: Adjust during training to minimize error.
   - **Activation Functions**: Introduce non-linearity.

5. **What is a Cost Function in FNN?**
   - The cost function measures the error between the predicted and actual values. It helps guide the learning process by adjusting weights to minimize this error.

6. **Mean Squared Error (MSE) Cost Function**
   - MSE is a common cost function that calculates the average of the squared differences between predicted and actual values.

7. **What is a Loss Function in FNN?**
   - A loss function calculates the error for a single training example, while the cost function is the average of losses across all examples.

8. **What is Cross-Entropy Loss?**
   - Cross-entropy loss is commonly used for classification tasks. It quantifies the difference between two probability distributions, often between true labels and predicted probabilities.

9. **Kernel Concept in FNN**
   - In neural networks, a "kernel" refers to the filter applied in convolutional neural networks (CNNs) rather than feedforward networks. However, FNNs may still use kernel techniques for learning patterns in data.

10. **MNIST and CIFAR-10 Datasets**
    - **MNIST**: A dataset of 28x28 grayscale images of handwritten digits (0-9), used for image classification tasks.
    - **CIFAR-10**: A dataset containing 60,000 32x32 color images across 10 classes (airplane, car, bird, etc.).

11. **SKlearn and TensorFlow/Keras Libraries for FNNs**
    - **LabelBinarizer**: Transforms labels into one-hot encoded vectors.
    - **classification_report**: Provides precision, recall, F1-score, and support metrics for evaluation.
    - **TensorFlow/Keras Libraries**:
      - **models**: Defines neural network models.
      - **layers**: Defines the layers in the network.
      - **optimizers**: Configures how the model learns.
      - **datasets**: Provides dataset loading and preprocessing tools.

12. **Flattening the Dataset**
    - Flattening transforms a multi-dimensional input (e.g., 28x28 image) into a 1D vector, making it suitable for input into a dense layer. It’s necessary for connecting layers in fully connected neural networks.

13. **Difference Between Sigmoid and Softmax Activation Functions**
    - **Sigmoid**: Used in binary classification; outputs values between 0 and 1.
    - **Softmax**: Used in multi-class classification; outputs probabilities for each class that sum to 1.

14. **Significance of Optimizer in Training Model**
    - Optimizers adjust the model’s weights based on the gradients of the loss function, aiding in the minimization of the error during training.

15. **Epochs in the `fit` Command**
    - An epoch represents one complete pass of the entire training dataset through the neural network. More epochs allow the model to learn better, but too many can lead to overfitting.

---

These answers provide a well-rounded preparation for implementing and understanding feedforward neural networks in both practical and viva contexts. Let me know if you’d like further clarification on any points!