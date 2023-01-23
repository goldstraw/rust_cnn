use rand_distr::{Normal, Distribution};

use crate::LEARNING_RATE;

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
    pub fn create_conv_layer(
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
    ) -> ConvLayer {
        let mut biases: Vec<f32> = vec![];
        let mut kernels: Vec<Vec<Vec<Vec<f32>>>> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for f in 0..num_filters {
            biases.push(normal.sample(&mut rand::thread_rng()));
            kernels.push(vec![]);
            for i in 0..input_depth {
                kernels[f].push(vec![]);
                for j in 0..kernel_size {
                    kernels[f][i].push(vec![]);
                    for _ in 0..kernel_size {
                        kernels[f][i][j].push(normal.sample(&mut rand::thread_rng()));
                    }
                }
            }
        }

        let output_size: usize = ((input_size - kernel_size) / stride) + 1;
        let mut output: Vec<Vec<Vec<f32>>> = vec![];
        for f in 0..num_filters {
            output.push(vec![]);
            for h in 0..output_size {
                output[f].push(vec![]);
                for _ in 0..output_size {
                    output[f][h].push(0.0);
                }
            }
        }

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
            output,
        };

        layer
    }

    pub fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        self.input = input.clone();
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                let left = x * self.stride;
                let top = y * self.stride;
                for f in 0..self.num_filters {
                    self.output[f][y][x] = self.biases[f];
                    for y_k in 0..self.kernel_size {
                        for x_k in 0..self.kernel_size {
                            for f_i in 0..self.input_depth {
                                let val: f32 = input[f_i][top + y_k][left + x_k];
                                self.output[f][y][x] += self.kernels[f][f_i][y_k][x_k] * val;
                            }
                        }
                    }
                }
            }
        }

        self.output.clone()
    }

    pub fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let mut prev_error: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0; self.input_size]; self.input_size]; self.input_depth];
        let mut new_kernels: Vec<Vec<Vec<Vec<f32>>>> = self.kernels.clone();

        for y in 0..self.output_size {
            for x in 0..self.output_size {
                let left = x * self.stride;
                let top = y * self.stride;
                for f in 0..self.num_filters {
                    if self.output[f][y][x] > 0.0 {
                        self.biases[f] -= error[f][y][x] * LEARNING_RATE;
                        for y_k in 0..self.kernel_size {
                            for x_k in 0..self.kernel_size {
                                for f_i in 0..self.input_depth {
                                    prev_error[f_i][top + y_k][left + x_k] +=
                                        self.kernels[f][f_i][y_k][x_k] * error[f][y][x];
                                    new_kernels[f][f_i][y_k][x_k] -= self.input[f_i][top + y_k]
                                        [left + x_k]
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

        prev_error
    }
}