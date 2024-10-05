# Rust Convolutional Neural Network from Scratch

This repository contains a Rust implementation of a Convolutional Neural Network (CNN) built from scratch. This repository provides code for training on the MNIST dataset, and the 50States10K dataset.

All machine learning code is written from scratch, however the `ndarray` crate is used for matrix operations. When tuned correctly, the network should reach 90+% accuracy within one minute on the MNIST dataset.

## Overview

The repository implements the following features:

- Convolutional, max pooling, and fully connected layers
- ReLU and Softmax activation functions
- Cross-entropy loss function
- SGD, Momentum, RMSProp, and Adam optimizers
- Dropout
- He initialization

## Usage

To run the demo of the CNN, place the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in a folder named `data`, and use the following command:

```
$ cargo run --release
```

This command will run a demo of the CNN and train it on the MNIST dataset.

## Further Reading

For more information about this project, read [my blog post on CNNs](https://charliegoldstraw.com/articles/cnn/).

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the `LICENSE` file for details.
