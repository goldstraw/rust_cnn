pub struct MaxPoolingLayer {
    input_size: usize,
    input_depth: usize,
    kernel_size: usize,
    output_size: usize,
    stride: usize,
    output: Vec<Vec<Vec<f32>>>,
    highest_index: Vec<Vec<Vec<[usize; 2]>>>,
}

impl MaxPoolingLayer {
    pub fn create_mxpl_layer(
        input_size: usize,
        input_depth: usize,
        kernel_size: usize,
        stride: usize,
    ) -> MaxPoolingLayer {
        let output_size: usize = ((input_size - kernel_size) / stride) + 1;

        let layer: MaxPoolingLayer = MaxPoolingLayer {
            input_size,
            input_depth,
            kernel_size,
            output_size,
            stride,
            output: vec![vec![vec![0.0; output_size]; output_size]; input_depth],
            highest_index: vec![vec![vec![[0, 0]; output_size]; output_size]; input_depth],
        };

        layer
    }

    pub fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                let left = x * self.stride;
                let top = y * self.stride;
                for f in 0..self.input_depth {
                    self.output[f][y][x] = -1.0;
                    for y_p in 0..self.kernel_size {
                        for x_p in 0..self.kernel_size {
                            let val: f32 = input[f][top + y_p][left + x_p];
                            if val > self.output[f][y][x] {
                                self.output[f][y][x] = val;
                                self.highest_index[f][y][x] = [top + y_p, left + x_p];
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

        for y in 0..self.output_size {
            for x in 0..self.output_size {
                for f in 0..self.input_depth {
                    let m: [usize; 2] = self.highest_index[f][y][x];
                    prev_error[f][m[0]][m[1]] = error[f][y][x];
                }
            }
        }

        prev_error
    }
}