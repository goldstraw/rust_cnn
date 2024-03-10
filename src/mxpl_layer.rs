use ndarray::{Array4, Array3};
use serde::{Serialize, Deserialize};
use std::fmt::{Debug, Formatter};

/// Defines a `MaxPoolingLayer` structure.
#[derive(Serialize, Deserialize)]
pub struct MxplLayer {
    input_size: (usize, usize, usize),
    kernel_size: usize,
    pub output_size: (usize, usize, usize),
    #[serde(skip)]
    highest_indices: Array4<usize>,
    stride: usize,
}

impl Debug for MxplLayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("Max Pooling Layer\n");
        s.push_str(&format!("Input Size: {}x{}x{}\n", self.input_size.0, self.input_size.1, self.input_size.2));
        s.push_str(&format!("Kernel Size: {}x{}\n", self.kernel_size, self.kernel_size));
        s.push_str(&format!("Output Size: {}x{}x{}\n", self.output_size.0, self.output_size.1, self.output_size.2));
        s.push_str(&format!("Stride: {}\n", self.stride));

        write!(f, "{}", s)
    }
}

impl MxplLayer {
    /// Create a new max pooling layer with the given parameters
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: usize,
        stride: usize,
    ) -> MxplLayer {
        let output_width: usize = ((input_size.0 - kernel_size) / stride) + 1;
        let output_size = (output_width, output_width, input_size.2);
        let layer: MxplLayer = MxplLayer {
            input_size,
            kernel_size,
            output_size,
            stride,
            highest_indices: Array4::<usize>::zeros((output_width, output_width, input_size.2, 2)),
        };

        layer
    }

    pub fn zero(&mut self) {
        self.highest_indices = Array4::<usize>::zeros((self.output_size.0, self.output_size.1, self.input_size.2, 2));
    }
    
    pub fn forward_propagate(&mut self, input: Array3<f32>) -> Array3<f32> {
        let mut output: Array3<f32> = Array3::<f32>::zeros(self.output_size);
        
        for f in 0..self.output_size.2 {
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    output[[x, y, f]] = -1.0;
                    self.highest_indices[[x, y, f, 0]] = 0;
                    self.highest_indices[[x, y, f, 1]] = 0;
                    
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            let index: (usize, usize) = (x * self.stride + kx, y * self.stride + ky);
                            let value: f32 = input[[index.0, index.1, f]];

                            if value > output[[x, y, f]] {
                                output[[x, y, f]] = value;
                                self.highest_indices[[x, y, f, 0]] = index.0;
                                self.highest_indices[[x, y, f, 1]] = index.1;
                            }
                        }
                    }
                }
            }
        }
        output
    }

    pub fn back_propagate(&mut self, error: Array3<f32>) -> Array3<f32> {
        let mut prev_error: Array3<f32> = Array3::<f32>::zeros(self.input_size);

        for f in 0..self.output_size.2 {
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    let hx: usize = self.highest_indices[[x, y, f, 0]];
                    let hy: usize = self.highest_indices[[x, y, f, 1]];
                    prev_error[[hx, hy, f]] = error[[x, y, f]];
                }
            }
        }

        prev_error
    }

    pub fn update(&mut self, _minibatch_size: usize) {}
}