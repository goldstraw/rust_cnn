use rand_distr::{Normal, Distribution};

use crate::{LEARNING_RATE, layer::Layer};

/// Defines a `ConvolutionalLayer` structure.
pub struct ConvLayer {
    input_size: usize,
    input_depth: usize,
    num_filters: usize,
    kernel_size: usize,
    output_size: usize,
    stride: usize,
    biases: Vec<f32>,
    kernels: Vec<Vec<Vec<Vec<f32>>>>,
    input: Vec<Vec<Vec<f32>>>,
    output: Vec<Vec<Vec<f32>>>,
}

impl ConvLayer {
    /// Creates a new convolutional layer with the given parameters
    pub fn new(
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
    ) -> ConvLayer {
        // Initialize the biases and kernels with empty vectors
        let mut biases = vec![];
        let mut kernels = vec![vec![vec![vec![]; kernel_size]; input_depth]; num_filters];

        // Use He initialisation by using a mean of 0.0 and a standard deviation of sqrt(2/(input_channels * num_params))
        let normal = Normal::new(0.0, (2.0/(input_depth*kernel_size.pow(2)) as f32).sqrt()).unwrap();

        // Fill the biases and kernels with random values from the normal distribution
        for f in 0..num_filters {
            biases.push(0.1);
            for i in 0..input_depth {
                for j in 0..kernel_size {
                    for _ in 0..kernel_size {
                        kernels[f][i][j].push(normal.sample(&mut rand::thread_rng()));
                    }
                }
            }
        }

        let output_size: usize = ((input_size - kernel_size) / stride) + 1;

        // Create the ConvolutionalLayer struct and return it
        let layer: ConvLayer = ConvLayer {
            input_size,
            input_depth,
            num_filters,
            kernel_size,
            output_size,
            stride,
            biases,
            kernels,
            input: vec![],
            output: vec![vec![vec![0.0; output_size]; output_size]; num_filters],
        };

        layer
    }
}

impl Layer for ConvLayer {

    /// Forward propagates the input data through the Convolutional layer.
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Store the input data in a member variable for future reference.
        self.input = input.clone();

        // Iterate through each output point in the output matrix.
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                // Calculate the starting point for the convolutional kernel.
                let left = x * self.stride;
                let top = y * self.stride;
                // Iterate through each filter in the network.
                for f in 0..self.num_filters {
                    // Initialize the output value with the bias value for the filter.
                    self.output[f][y][x] = self.biases[f];
                    
                    // Iterate through each input channel.
                    for f_i in 0..self.input_depth {
                        for y_k in 0..self.kernel_size {
                            for x_k in 0..self.kernel_size {
                                // Get the value of the input at the current point.
                                let val: f32 = input[f_i][top + y_k][left + x_k];
                                // Store the result of the convolution in the output matrix.
                                self.output[f][y][x] += self.kernels[f][f_i][y_k][x_k] * val;
                            }
                        }
                    }
                }
            }
        }

        // Apply the ReLU activation function to the output.
        for f in 0..self.num_filters {
            for y in 0..self.output_size {
                for x in 0..self.output_size {
                    self.output[f][y][x] = self.output[f][y][x].max(0.0);
                }
            }
        }

        self.output.clone()
    }

    /// Back propagates the error through the Convolutional layer.
    /// Returns the error for the previous layer.
    fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let mut prev_error: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0; self.input_size]; self.input_size]; self.input_depth];
        let mut new_kernels: Vec<Vec<Vec<Vec<f32>>>> = self.kernels.clone();

        // Iterate through each output point in the output matrix.
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                // Calculate the receptive field coordinates for the current output point.
                let left = x * self.stride;
                let top = y * self.stride;
                // Iterate through each filter in the network.
                for f in 0..self.num_filters {
                    // Only update parameters which affect the output.
                    if self.output[f][y][x] > 0.0 {
                        self.biases[f] -= error[f][y][x] * LEARNING_RATE;
                        for y_k in 0..self.kernel_size {
                            for x_k in 0..self.kernel_size {
                                for f_i in 0..self.input_depth {
                                    // Update the error for the previous layer.
                                    prev_error[f_i][top + y_k][left + x_k] +=
                                        self.kernels[f][f_i][y_k][x_k] * error[f][y][x];
                                    // Store the new kernel values.
                                    new_kernels[f][f_i][y_k][x_k] -= self.input[f_i][top + y_k][left + x_k]
                                        * error[f][y][x]
                                        * LEARNING_RATE;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.kernels = new_kernels;

        // Return the error for the previous layer.
        prev_error
    }

    fn get_output(&mut self, _index: usize) -> f32 {
        panic!("Convolutional layers should not be accessed directly.")
    }
}