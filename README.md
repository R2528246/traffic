[README (1).md](https://github.com/user-attachments/files/35482405/README.1.md)
# Traffic Sign Classification

## Project Overview

This project uses a neural network to classify German traffic sign images into 43 different categories using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Files

- `traffic.py`: Contains the image-loading function, neural network model and training process.
- `gtsrb/`: Contains the traffic sign image dataset.

## How the Program Works

The program loads traffic sign images, resizes them to the required dimensions, assigns their corresponding labels, and uses a convolutional neural network to classify the images.

The project uses a Sequential Convolutional Neural Network (CNN).

The architecture consists of:

1. An input layer accepting RGB images of size 30 × 30 × 3.
2. A rescaling layer that normalizes pixel values to the range 0–1.
3. A convolutional layer with 32 filters and ReLU activation.
4. A max-pooling layer with a 2 × 2 pool size.
5. A second convolutional layer with 64 filters and ReLU activation.
6. A second max-pooling layer with a 2 × 2 pool size.
7. A flattening layer to convert feature maps into a one-dimensional vector.
8. A dense layer containing 128 neurons with ReLU activation.
9. A dropout layer with a rate of 0.5 to reduce overfitting.
10. An output layer with 43 neurons and softmax activation, representing the 43 traffic sign categories.

The model uses the Adam optimizer, categorical cross-entropy loss and accuracy as its evaluation metric.

## Experiments
The initial model uses two convolutional layers, with 32 and 64 filters respectively, followed by pooling layers and a dense classification layer.

Further experiments will compare alternative numbers of convolutional filters, dense-layer sizes and dropout rates.

The effect of each change will be evaluated using training and testing accuracy. The best-performing architecture will be selected based on its classification performance.

## Results

Add the final test accuracy and explain which architecture performed best after completing the experiments.
