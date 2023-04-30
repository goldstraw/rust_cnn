# Rust Convolutional Neural Network from Scratch

This repository contains a Rust implementation of a Convolutional Neural Network (CNN) built from scratch. The CNN is designed to learn and classify the MNIST dataset.

## Overview

The repository contains the following main components:

```
src/
├── cnn_struct.rs
├── conv_layer.rs
├── fully_connected_layer.rs
├── layer.rs
├── lib.rs
├── main.rs
├── max_pooling_layer.rs
└── run.rs
```

* `cnn_struct.rs`: Defines the structure of the CNN model.
* `conv_layer.rs`: Implements the convolutional layer for the CNN.
* `fully_connected_layer.rs`: Implements the fully connected layer for the CNN.
* `layer.rs`: Defines the interface for the CNN layers.
* `lib.rs`: The Rust library file.
* `main.rs`: A demo of the CNN's use.
* `max_pooling_layer.rs`: Implements the max pooling layer for the CNN.
* `run.rs`: Contains functions to run the CNN.

## Installation

To use this CNN implementation, you must have Rust and Cargo installed on your machine. After installing Rust and Cargo, you can clone this repository to your local machine and build the project with the following command:

```
$ cargo build
```

## Usage

To run the demo of the CNN, use the following command:

```
$ cargo run
```

This command will run a demo of the CNN and train it on the MNIST dataset.

## Further Reading

For more information about this project, read [my blog post on CNNs](https://charliegoldstraw.com/articles/cnn/).

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the `LICENSE` file for details.
