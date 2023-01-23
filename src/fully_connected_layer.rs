use rand_distr::{Normal, Distribution};

use crate::LEARNING_RATE;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn inv_deriv_sigmoid(x: f32) -> f32 {
    let z: f32 = (x / (1.0 - x)).ln();
    sigmoid(z) * (1.0 - sigmoid(z))
}

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
    pub fn create_fcl_layer(
        input_width: usize,
        input_depth: usize,
        output_size: usize,
    ) -> FullyConnectedLayer {
        let input_size: usize = input_depth * (input_width * input_width);
        let mut biases: Vec<f32> = vec![];
        let mut weights: Vec<Vec<f32>> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for i in 0..input_size {
            biases.push(normal.sample(&mut rand::thread_rng()));
            weights.push(vec![]);
            for _ in 0..output_size {
                weights[i].push(normal.sample(&mut rand::thread_rng()));
            }
        }

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

    pub fn forward_propagate(&mut self, input: Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        self.input = input.clone();
        for j in 0..self.output_size {
            self.output[j] = self.biases[j];
            for i in 0..self.input_size {
                self.output[j] += input[i] * self.weights[i][j];
            }
            self.output[j] = sigmoid(self.output[j]);
        }
        let formatted_output: Vec<Vec<Vec<f32>>> = vec![vec![self.output.clone()]];
        formatted_output
    }

    pub fn back_propagate(&mut self, error: Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        let mut flat_error: Vec<f32> = vec![0.0; self.input_size];

        for j in 0..self.output_size {
            self.biases[j] -= error[j] * LEARNING_RATE;
            for i in 0..self.input_size {
                flat_error[i] += error[j] * self.weights[i][j];
                self.weights[i][j] -=
                    error[j] * self.input[i] * inv_deriv_sigmoid(self.output[j]) * LEARNING_RATE;
            }
        }

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