use rand_distr::{Distribution, Normal};

use crate::LEARNING_RATE;

/// Sigmoid activation function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse derivative of the sigmoid function
pub fn inv_deriv_sigmoid(x: f32) -> f32 {
    let z: f32 = (x / (1.0 - x)).ln();
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// Defines a `FullyConnectedLayer` structure.
pub struct FullyConnectedLayer {
    input_size: usize,
    input_width: usize,
    input_depth: usize,
    output_size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    input: Vec<f32>,
    pub output: Vec<f32>,
}

impl FullyConnectedLayer {
    /// Creates a new fully connected layer with the given input width, input depth, and output
    pub fn new(
        input_width: usize,
        input_depth: usize,
        output_size: usize,
    ) -> FullyConnectedLayer {
        // Calculate the input size from the input width and depth
        let input_size: usize = input_depth * (input_width * input_width);
        // Initialize empty vectors for the biases and weights
        let mut biases: Vec<f32> = vec![];
        let mut weights: Vec<Vec<f32>> = vec![vec![]; input_size];
        // Use He initialisation by using a mean of 0.0 and a standard deviation of sqrt(2/input_neurons)
        let normal = Normal::new(0.0, (2.0/(input_size.pow(2) * input_depth) as f32).sqrt()).unwrap();
        
        for _ in 0..output_size {
            // Initialize the biases with a small positive value
            biases.push(0.1);
            for i in 0..input_size {
                // Initialize the weights with random values drawn from the normal distribution
                weights[i].push(normal.sample(&mut rand::thread_rng()));
            }
        }

        // Create and return a new FullyConnectedLayer with the initialized values
        let layer: FullyConnectedLayer = FullyConnectedLayer {
            input_size,
            input_width,
            input_depth,
            output_size,
            weights,
            biases,
            input: vec![],
            output: vec![0.0; output_size],
        };

        layer
    }

    /// Calculates the output layer by forward propagating the input layer.
    pub fn forward_propagate(&mut self, input: Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        // Store the input for backpropagation
        self.input = input.clone();
        for j in 0..self.output_size {
            // Calculate the weighted sum of the inputs
            self.output[j] = self.biases[j];
            for i in 0..self.input_size {
                self.output[j] += input[i] * self.weights[i][j];
            }
            // Apply the sigmoid activation function to the output
            self.output[j] = sigmoid(self.output[j]);
        }

        // Format the output to be a 3D vector
        let formatted_output: Vec<Vec<Vec<f32>>> = vec![vec![self.output.clone()]];
        formatted_output
    }

    /// Back propagates the error through the fully connected layer.
    /// Updates the weights and biases of the current layer.
    /// Returns the error of the previous layer.
    pub fn back_propagate(&mut self, error: Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        let mut flat_error: Vec<f32> = vec![0.0; self.input_size];

        // Update the weights and biases according to their derivatives
        for j in 0..self.output_size {
            self.biases[j] -= error[j] * LEARNING_RATE;
            for i in 0..self.input_size {
                flat_error[i] += error[j] * self.weights[i][j];
                self.weights[i][j] -=
                    error[j] * self.input[i] * inv_deriv_sigmoid(self.output[j]) * LEARNING_RATE;
            }
        }

        // Format the error to be a 3D vector
        let mut prev_error: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![]; self.input_width]; self.input_depth];
        for i in 0..self.input_depth {
            for j in 0..self.input_width {
                for k in 0..self.input_width {
                    let index: usize = i * self.input_width.pow(2) + j * self.input_width + k;
                    prev_error[i][j].push(flat_error[index]);
                }
            }
        }

        prev_error
    }
}
