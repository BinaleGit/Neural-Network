import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset using TensorFlow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images to fit the model (28x28 pixels to 784)
train_images = train_images.reshape((train_images.shape[0], 784))
test_images = test_images.reshape((test_images.shape[0], 784))

# Initialize model parameters
input_size = 784
hidden_size = 64
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(Y, A2):
    m = Y.shape[0]
    logprobs = -np.log(A2[range(m), np.argmax(Y, axis=1)])
    loss = np.sum(logprobs) / m
    return loss

def backward_propagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[0]
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train(X, Y, learning_rate=0.1, epochs=100):
    global W1, b1, W2, b2
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X)
        loss = compute_loss(Y, A2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# One-hot encode the labels
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

train_labels_encoded = one_hot_encode(train_labels, 10)
test_labels_encoded = one_hot_encode(test_labels, 10)

# Train the model
train(train_images, train_labels_encoded, learning_rate=0.1, epochs=100)

# Predict function to evaluate on test set
def predict(X):
    _, _, _, A2 = forward_propagation(X)
    return np.argmax(A2, axis=1)

# Predict on the test set
predicted_labels = predict(test_images)

# Display a few test images along with the predicted and actual labels
def display_images(images, actual_labels, predicted_labels, num_images=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}")
        plt.axis('off')
    plt.show()

# Show the first 5 images from the test set along with their predicted and actual labels
display_images(test_images, test_labels, predicted_labels, num_images=5)
