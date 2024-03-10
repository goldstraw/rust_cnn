use std::ops::{AddAssign, SubAssign};
use std::fmt::{Debug, Formatter};
use ndarray::{Array3, Array4, s};
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};
use crate::optimizer::{Optimizer4D, OptimizerAlg};

#[derive(Serialize, Deserialize)]
pub struct ConvLayer {
    input_size: (usize, usize, usize),
    kernel_size: usize,
    pub output_size: (usize, usize, usize),
    #[serde(skip)]
    input: Array3<f32>,
    #[serde(skip)]
    output: Array3<f32>,
    stride: usize,
    num_filters: usize,
    kernels: Array4<f32>,
    #[serde(skip)]
    kernel_changes: Array4<f32>,
    optimizer: Optimizer4D,
}

impl Debug for ConvLayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("Convolutional Layer\n");
        s.push_str(&format!("Input Size: {}x{}x{}\n", self.input_size.0, self.input_size.1, self.input_size.2));
        s.push_str(&format!("Kernel Size: {}x{}\n", self.kernel_size, self.kernel_size));
        s.push_str(&format!("Output Size: {}x{}x{}\n", self.output_size.0, self.output_size.1, self.output_size.2));
        s.push_str(&format!("Stride: {}\n", self.stride));
        s.push_str(&format!("Number of Filters: {}\n", self.num_filters));

        write!(f, "{}", s)
    }
}

impl ConvLayer {
    pub fn zero(&mut self) {
        self.kernel_changes = Array4::<f32>::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
        self.output = Array3::<f32>::zeros(self.output_size);
    }

    /// Create a new max pooling layer with the given parameters
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: usize,
        stride: usize,
        num_filters: usize,
        optimizer_alg: OptimizerAlg,
    ) -> ConvLayer {
        let output_width: usize = ((input_size.0 - kernel_size) / stride) + 1;
        let output_size = (output_width, output_width, num_filters);
        let mut kernels = Array4::<f32>::zeros((num_filters, kernel_size, kernel_size, input_size.2));
        let normal = Normal::new(0.0, 1.0).unwrap();

        for f in 0..num_filters {
            for kd in 0..input_size.2 {
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        kernels[[f, ky, kx, kd]] = normal.sample(&mut rand::thread_rng()) * (2.0/(input_size.0.pow(2)) as f32).sqrt();
                    }
                }
            }
        }

        let optimizer = Optimizer4D::new(optimizer_alg, (num_filters, kernel_size, kernel_size, input_size.2));
        
        let layer: ConvLayer = ConvLayer {
            input_size,
            kernel_size,
            output_size,
            stride,
            output: Array3::<f32>::zeros(output_size),
            input: Array3::<f32>::zeros(input_size),
            num_filters,
            kernels,
            kernel_changes: Array4::<f32>::zeros((num_filters, kernel_size, kernel_size, input_size.2)),
            optimizer,
        };
        
        layer
    }

    pub fn forward_propagate(&mut self, input: Array3<f32>) -> Array3<f32> {
        self.input = input;
        for f in 0..self.output_size.2 {
            let kernel_slice = self.kernels.slice(s![f, .., .., ..]);
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    let input_slice = self.input.slice(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]);
                    self.output[[x, y, f]] = (&input_slice * &kernel_slice).sum().max(0.0);
                }
            }
        }

        self.output.clone()
    }

    pub fn back_propagate(&mut self, error: Array3<f32>) -> Array3<f32> {
        let mut prev_error: Array3<f32> = Array3::<f32>::zeros(self.input_size);
        for f in 0..self.output_size.2 {
            for y in 0..self.output_size.1 {
                for x in 0..self.output_size.0 {
                    if self.output[[x, y, f]] <= 0.0 {
                        continue;
                    }
                    prev_error.slice_mut(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]).add_assign(&(error[[x, y, f]] * &self.kernels.slice(s![f, .., .., ..])));
                    
                    let input_slice = self.input.slice(s![x..x+self.kernel_size, y..y+self.kernel_size, ..]);
                    self.kernel_changes.slice_mut(s![f, .., .., ..]).sub_assign(&(error[[x, y, f]] * &input_slice));
                }
            }
        }

        prev_error
    }

    pub fn update(&mut self, minibatch_size: usize) {
        self.kernel_changes /= minibatch_size as f32;
        self.kernels += &self.optimizer.weight_changes(&self.kernel_changes);
        self.kernel_changes = Array4::<f32>::zeros((self.num_filters, self.kernel_size, self.kernel_size, self.input_size.2));
    }
}