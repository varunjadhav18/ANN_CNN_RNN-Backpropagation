# Neural Network Backpropagation (ANN, CNN, RNN)

This repository demonstrates **how backpropagation works internally** in three types of neural networks:

* Artificial Neural Network (ANN)
* Convolutional Neural Network (CNN)
* Recurrent Neural Network (RNN)

Unlike typical implementations, this project explicitly prints:

* Predictions
* Loss values
* Gradients
* Weights **before and after updates**

This helps in understanding *how learning actually happens*.

---

# 📌 1. Artificial Neural Network (ANN)

## Overview

The ANN implemented here is a simple **2-layer feedforward network** trained on a small dataset.

### Architecture

* Input layer: 2 neurons
* Hidden layer: 3 neurons (Sigmoid activation)
* Output layer: 1 neuron (Sigmoid)

---

## Forward Pass

Steps:

1. Compute hidden layer:
   z1 = X · W1 + b1
   a1 = sigmoid(z1)

2. Compute output:
   z2 = a1 · W2 + b2
   y_pred = sigmoid(z2)

---

## Loss Function

Mean Squared Error (MSE):

Loss = mean((y - y_pred)²)

---

## Backpropagation

Gradients are computed using the chain rule:

* Output layer:
  dL/dz2 = (y_pred - y) * sigmoid_derivative(y_pred)

* Hidden layer:
  dL/dz1 = (dL/dz2 · W2ᵀ) * sigmoid_derivative(a1)

---

## Weight Updates

Weights are updated using gradient descent:

W = W - learning_rate * gradient

---

## Output Explanation

Each epoch prints:

* Loss value
* Predictions
* Weights before update
* Weights after update

---

# 📌 2. Convolutional Neural Network (CNN)

## Overview

This is a minimal CNN using PyTorch to demonstrate backpropagation through:

* Convolution layer
* Fully connected layer

---

## Architecture

* Conv2D (1 filter, kernel size 2×2)
* ReLU activation
* Flatten layer
* Fully connected layer
* Sigmoid output

---

## Forward Pass

1. Apply convolution
2. Apply ReLU activation
3. Flatten output
4. Fully connected layer
5. Sigmoid activation

---

## Backpropagation

PyTorch automatically computes gradients using **autograd**, but we explicitly print them:

* `model.conv.weight.grad` → gradient of convolution filters

---

## Output Explanation

Each epoch prints:

* Loss
* Prediction
* Convolution weights (before update)
* Gradients of weights
* Convolution weights (after update)

---

## Key Insight

CNN backpropagation updates **filters (kernels)** instead of simple weight matrices.

---

# 📌 3. Recurrent Neural Network (RNN)

## Overview

This example demonstrates **Backpropagation Through Time (BPTT)** using a simple RNN.

---

## Architecture

* RNN layer (hidden size = 2)
* Fully connected output layer
* Sigmoid activation

---

## Forward Pass

1. Input sequence is passed through RNN
2. Hidden states are computed at each time step
3. Only the **last hidden state** is used for prediction

---

## Backpropagation Through Time (BPTT)

Unlike ANN/CNN:

* Gradients are propagated **across time steps**
* Each time step contributes to the final loss

---

## Output Explanation

Each epoch prints:

* Loss
* Prediction
* RNN weights before update
* Gradients
* Updated weights

---

## Key Insight

RNN learns by adjusting weights based on:

* Current input
* Previous hidden states

---

# 🔍 What Makes This Project Special

✔ Shows **internal learning mechanics**
✔ Prints **real weight updates**
✔ Displays **gradients explicitly**
✔ Covers **three major neural network types**

---

# 📊 Example Outputs

You will see outputs like:

* Loss decreasing over epochs
* Predictions getting closer to actual values
* Gradients indicating direction of learning
* Weights being updated step-by-step

---

# 🚀 How to Run

## Requirements

* Python 3.x
* NumPy
* PyTorch

Install dependencies:

```bash
pip install numpy torch
```

Run the scripts:

```bash
python ann.py
python cnn.py
python rnn.py
```

---

# 🧠 Learning Outcomes

After running this code, you will understand:

* How forward propagation works
* How gradients are computed
* How weights are updated
* Difference between ANN, CNN, and RNN training

---

# 📌 Summary

| Model | Key Feature            | Backpropagation Type  |
| ----- | ---------------------- | --------------------- |
| ANN   | Fully connected layers | Standard backprop     |
| CNN   | Convolution filters    | Spatial backprop      |
| RNN   | Sequential data        | Backprop Through Time |

---

# 📚 Conclusion

This project is designed for **deep understanding**, not just usage.

If you want to truly understand neural networks, inspecting:

* weights
* gradients
* predictions

is essential — and this repo gives you exactly that.

---
