# ML-Experimentation: MNIST Neural Network with PyTorch

This project implements a simple feedforward neural network to classify handwritten digits from the MNIST dataset using PyTorch.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the neural network training and testing:
   ```
   python mnist_nn.py
   ```

## What it does

- Downloads the MNIST dataset automatically (if not already present)
- Trains a simple neural network with 2 hidden layers
- Tests the model on the test set
- Saves the trained model as `mnist_model.pth`

## Dataset

The MNIST dataset is the "hello world" of machine learning, consisting of 70,000 grayscale images of handwritten digits (0-9).

- Official website: http://yann.lecun.com/exdb/mnist/
- In this code, we use PyTorch's `torchvision.datasets.MNIST` which downloads the dataset automatically.

## Troubleshooting

- If you encounter CUDA-related errors, the code will fall back to CPU training.
- Ensure you have Python 3.7+ installed.
- For GPU acceleration, install PyTorch with CUDA support if you have an NVIDIA GPU.