# Handwritten Digit Recognizer (神经网络)

**📄 License: Free to Use** - This project is freely available for educational and personal use.

A neural network implementation built from scratch using NumPy to recognize handwritten digits from the MNIST dataset. This project demonstrates core deep learning concepts including forward propagation, backpropagation, activation functions, and optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technologies & Dependencies](#technologies--dependencies)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [How It Works](#how-it-works)
- [Key Mathematical Concepts](#key-mathematical-concepts)
- [Usage](#usage)
- [Learning Outcomes](#learning-outcomes)
- [Future Enhancements](#future-enhancements)
- [Dataset](#dataset)
- [References](#references)
- [License](#license)

## Overview

This is an educational implementation of a neural network designed to classify handwritten digits (0-9) with approximately **93% accuracy** after 400 training epochs. The entire model is implemented from first principles using only NumPy and idx2numpy, without relying on deep learning frameworks like TensorFlow or PyTorch.

**Key Performance Metrics:**
- Accuracy: ~93%
- Training Epochs: 400
- Dataset: MNIST (60,000 training images, 10,000 test images)
- Input Size: 784 pixels (28×28 images)
- Output Classes: 10 (digits 0-9)

## Architecture

The neural network consists of:
- **Input Layer**: 784 neurons (flattened 28×28 pixel images)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one per digit)

## Technologies & Dependencies

- **Python 3.x**
- **NumPy**: Numerical computing and matrix operations
- **idx2numpy**: For loading and converting MNIST IDX binary format

Install dependencies:
```bash
pip install numpy idx2numpy
```

## Project Structure

```
digit-recognizer/
├── HDR.py              # Core neural network implementation
├── train.py            # Training script
├── README.md           # This file
└── dataset/
    ├── train-images.idx3-ubyte    # Training images
    ├── train-labels.idx1-ubyte    # Training labels
    ├── t10k-images.idx3-ubyte     # Test images
    └── t10k-labels.idx1-ubyte     # Test labels
```

## Core Components

### HDR.py - Neural Network Implementation

#### `data_handling(pathimages, pathlabels)`
Loads MNIST data from IDX binary format and normalizes pixel values to [0, 1] range.

#### `Layer` Class
Implements a fully connected neural network layer with:
- **Forward Propagation**: `fpropagation(input)` - Computes weighted sum and bias
- **Backward Propagation**: `backward(backwardpass)` - Calculates gradients for weights, biases, and inputs
- Weights initialized with small random values for stability

#### `Activation` Class
Implements ReLU (Rectified Linear Unit) activation function:
- **Forward**: `max(0, x)` - Non-linear activation
- **Backward**: Gradient passing for backpropagation

#### `Softmax` Class
Converts raw network output to probability distribution:
- Numerically stable implementation with max subtraction
- Output sums to 1 across all classes

#### `Loss` Class
Computes categorical cross-entropy loss:
- Formula: $-\ln(y_{pred})$ for the correct class
- Clipping predictions to prevent log(0) errors

#### `Backpropagation` Class
Combines Softmax and Loss for efficient gradient calculation:
- Integrates forward pass through activation and loss
- Computes backward gradients for optimization

#### `Optimizer` Class
Implements basic Stochastic Gradient Descent (SGD):
- Adjusts weights and biases using learning rate
- Simple parameter update rule: $\theta = \theta - \alpha \cdot \nabla\theta$

### train.py - Training Script

Trains the neural network with:
- Configurable learning rate (0.5)
- 400 training epochs
- Periodic loss and accuracy logging (every 100 epochs)
- Full forward and backward passes for each epoch

## How It Works

### Forward Propagation
1. Input images (784 features) pass through hidden layer 1
2. ReLU activation introduces non-linearity
3. Output passes through hidden layer 2 with ReLU
4. Final layer output fed through Softmax to get probabilities
5. Loss computed using categorical cross-entropy

### Backward Propagation
1. Compute gradient of loss with respect to output
2. Propagate gradients back through Softmax and Loss
3. Propagate through second hidden layer and its activation
4. Propagate through first hidden layer and its activation
5. Update all weights and biases using SGD optimizer

### Key Mathematical Concepts

**ReLU Activation**: 
$$f(x) = \max(0, x)$$

**Softmax Function**:
$$\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Cross-Entropy Loss**:
$$L = -\sum_i y_i \ln(\hat{y}_i)$$

where $y_i$ is the true label and $\hat{y}_i$ is the predicted probability.

## Usage

### Training the Model

```bash
python train.py
```

The script will:
1. Load training data from the dataset folder
2. Initialize the neural network with three layers
3. Train for 400 epochs
4. Print loss and accuracy every 100 epochs

Expected output:
```
Epoch: 0, Loss: 2.297, Accuracy: 0.107
Epoch: 100, Loss: 0.218, Accuracy: 0.935
Epoch: 200, Loss: 0.143, Accuracy: 0.956
Epoch: 300, Loss: 0.115, Accuracy: 0.963
```

## Learning Outcomes

This project teaches:
- Neural network fundamentals from scratch
- Forward and backward propagation mechanics
- Activation functions (ReLU, Softmax)
- Loss functions and optimization
- Matrix operations and NumPy usage
- MNIST dataset handling

## Future Enhancements

- **Custom Digit Recognition System**: An end-to-end system allowing users to input custom digits (28×28 pixel images) and receive predictions with confidence scores
- Test set evaluation script
- Visualization of learned weights
- Hyperparameter tuning utilities
- Additional activation functions (Sigmoid, Tanh)
- Batch normalization
- Regularization techniques (L1/L2)
- Model saving and loading capabilities

## Dataset

The MNIST dataset consists of 70,000 handwritten digit images:
- **60,000 training images**
- **10,000 test images**
- Image size: 28×28 pixels (784 features when flattened)
- Pixel values: 0-255 (normalized to 0-1)

Dataset source: [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## References

- [ReLU Activation Function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Softmax Function](https://en.wikipedia.org/wiki/Softmax_function)
- [Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [idx2numpy GitHub](https://github.com/ivanyu/idx2numpy)

## License

**Free to Use** - This educational project is provided as-is for learning purposes. You are free to use, modify, and distribute this project for educational and personal projects.

## Author

Created as an educational resource to demonstrate neural network principles using pure Python and NumPy.


