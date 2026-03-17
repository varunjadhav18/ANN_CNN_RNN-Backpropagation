# Backpropagation in ANN, CNN, and RNN

This repository demonstrates how **Backpropagation** works in three major neural network architectures:

- Artificial Neural Network (**ANN**)
- Convolutional Neural Network (**CNN**)
- Recurrent Neural Network (**RNN**)

The project explains the **mathematical formulas, algorithms, and gradient calculations** used during training.

---

# Table of Contents

1. Introduction  
2. Backpropagation Overview  
3. Artificial Neural Network (ANN)  
4. Convolutional Neural Network (CNN)  
5. Recurrent Neural Network (RNN)  
6. Comparison  
7. Requirements  
8. How to Run  

---

# 1. Introduction

Neural networks learn by adjusting weights to minimize prediction error.  
This process is done using **Backpropagation with Gradient Descent**.

The objective is to minimize the loss function:

L = (1/N) Σ (y − ŷ)²

Where:

- y = true output  
- ŷ = predicted output  
- N = number of samples  

---

# 2. Backpropagation Overview

Backpropagation has two phases:

## Forward Propagation
The input passes through layers to generate predictions.

## Backward Propagation
The error is propagated backward to update weights.

Weight update rule:

W_new = W − η * (∂L / ∂W)

Where:

- W = weight  
- η = learning rate  
- L = loss function  

---

# 3. Artificial Neural Network (ANN)

## Architecture

Input Layer → Hidden Layer → Output Layer

---

## Forward Propagation

Hidden layer calculation:

Z1 = W1X + b1

Activation function:

A1 = f(Z1)

Output layer:

Z2 = W2A1 + b2

Prediction:

ŷ = f(Z2)

---

## Loss Function

Mean Squared Error:

L = 1/2 (y − ŷ)²

---

## Backpropagation Calculations

Output layer error:

δ2 = (ŷ − y) * f'(Z2)

Gradient for output weights:

∂L/∂W2 = δ2 * A1

Hidden layer error:

δ1 = (W2 * δ2) * f'(Z1)

Gradient for hidden weights:

∂L/∂W1 = δ1 * X

---

## ANN Training Algorithm

1. Initialize weights randomly  
2. Perform forward propagation  
3. Compute loss  
4. Calculate gradients using backpropagation  
5. Update weights using gradient descent  
6. Repeat until convergence  

---

# 4. Convolutional Neural Network (CNN)

CNNs are mainly used for **image data processing**.

---

## Architecture

Input Image  
→ Convolution Layer  
→ Activation (ReLU)  
→ Pooling Layer  
→ Fully Connected Layer  
→ Output

---

## Convolution Operation

Feature map calculation:

F(i,j) = Σ Σ X(i+k, j+l) * K(k,l)

Where:

- X = input image  
- K = convolution filter  

---

## Activation Function

ReLU:

f(x) = max(0, x)

---

## Loss Function

Cross Entropy Loss:

L = − Σ y log(ŷ)

---

## Backpropagation in CNN

Gradient of filter:

∂L/∂K = X * δ

Gradient of input:

∂L/∂X = δ * K

Where:

- δ = error gradient  
- * = convolution operation  

---

## CNN Training Algorithm

1. Initialize convolution filters  
2. Perform convolution operation  
3. Apply activation function  
4. Apply pooling  
5. Flatten feature maps  
6. Feed into fully connected layer  
7. Compute loss  
8. Backpropagate gradients through layers  
9. Update filters and weights  

---

# 5. Recurrent Neural Network (RNN)

RNNs process **sequential data** such as text, speech, and time series.

---

## Architecture

At time step t:

Input → Hidden State → Output  
        ↑  
     Previous Hidden State

---

## Hidden State Calculation

h_t = f(Wx * x_t + Wh * h_(t-1) + b)

Where:

- x_t = input at time t  
- h_t = hidden state  
- Wx = input weight matrix  
- Wh = recurrent weight matrix  

---

## Output Calculation

y_t = Wy * h_t

---

## Loss Function

L = Σ (y_t − ŷ_t)²

---

## Backpropagation Through Time (BPTT)

Error at time t:

δ_t = (ŷ_t − y_t) * f'(h_t)

Gradient for recurrent weights:

∂L/∂Wh = Σ δ_t * h_(t-1)

Gradient for input weights:

∂L/∂Wx = Σ δ_t * x_t

---

## RNN Training Algorithm

1. Initialize weights  
2. Perform forward propagation through time  
3. Compute total loss  
4. Apply Backpropagation Through Time (BPTT)  
5. Compute gradients for each timestep  
6. Update weights using gradient descent  

---

# 6. Comparison

| Model | Data Type | Backpropagation Method |
|------|-----------|------------------------|
| ANN | Tabular Data | Standard Backpropagation |
| CNN | Image Data | Convolution Backpropagation |
| RNN | Sequential Data | Backpropagation Through Time |

---

# 7. Requirements

Python 3.x

Required libraries:

- numpy
- torch

Install dependencies:

pip install numpy torch

---

# 8. How to Run

Clone the repository:

git clone https://github.com/yourusername/backpropagation-models

Move into project folder:

cd backpropagation-models

Run ANN:

python ann_backpropagation.py

Run CNN:

python cnn_backpropagation.py

Run RNN:

python rnn_backpropagation.py

---

# Project Structure

backpropagation-models/

ann_backpropagation.py  
cnn_backpropagation.py  
rnn_backpropagation.py  
README.md

---

# References

- Deep Learning – Ian Goodfellow  
- Stanford CS231n Notes  
- Pattern Recognition and Machine Learning – Christopher Bishop  

---
