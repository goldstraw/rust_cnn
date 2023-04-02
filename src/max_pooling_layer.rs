use crate::layer::Layer;

/// Defines a `MaxPoolingLayer` structure.
pub struct MaxPoolingLayer {
    input_size: usize,
    input_depth: usize,
    kernel_size: usize,
    output_size: usize,
    stride: usize,
    output: Vec<Vec<Vec<f32>>>,
    highest_index: Vec<Vec<Vec<(usize, usize)>>>,
}

impl MaxPoolingLayer {
    /// Create a new max pooling layer with the given parameters
    pub fn new(
        input_size: usize,
        input_depth: usize,
        kernel_size: usize,
        stride: usize,
    ) -> MaxPoolingLayer {
        let output_size: usize = ((input_size - kernel_size) / stride) + 1;

        // Create a new MaxPoolingLayer with the initialized parameters and vectors
        let layer: MaxPoolingLayer = MaxPoolingLayer {
            input_size,
            input_depth,
            kernel_size,
            output_size,
            stride,
            output: vec![vec![vec![0.0; output_size]; output_size]; input_depth],
            highest_index: vec![vec![vec![(0, 0); output_size]; output_size]; input_depth],
        };

        layer
    }
}

impl Layer for MaxPoolingLayer {
    /// Reduces the size of the input by using max pooling.
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Loop through each output position in the output volume
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                // Calculate the top-left corner of the receptive field
                let left = x * self.stride;
                let top = y * self.stride;
                for f in 0..self.input_depth {
                    self.output[f][y][x] = -1.0;
                    // Loop through each position in the receptive field
                    // and find the highest value
                    for y_p in 0..self.kernel_size {
                        for x_p in 0..self.kernel_size {
                            let val: f32 = input[f][top + y_p][left + x_p];
                            if val > self.output[f][y][x] {
                                self.output[f][y][x] = val;

                                // Store the position of the highest value for backpropagation
                                self.highest_index[f][y][x] = (top + y_p, left + x_p);
                            }
                        }
                    }
                }
            }
        }
        self.output.clone()
    }

    /// Back propagates the error in a max pooling layer. 
    /// Takes in the error matrix and returns the previous error matrix
    fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let mut prev_error: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0; self.input_size]; self.input_size]; self.input_depth];

        // Iterate through the output neurons
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                // Input depth will always be the same as output depth
                for f in 0..self.input_depth {
                    let m: (usize, usize) = self.highest_index[f][y][x];
                    // Update the input error value with the corresponding output error value
                    prev_error[f][m.0][m.1] = error[f][y][x];
                }
            }
        }

        // Return the previous error vector
        prev_error
    }

    fn get_output(&mut self, _index: usize) -> f32 {
        panic!("Max pooling layers should not be accessed directly.")
    }
}