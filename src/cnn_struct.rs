use crate::{
    conv_layer::ConvLayer, fully_connected_layer::FullyConnectedLayer, layer::Layer,
    max_pooling_layer::MaxPoolingLayer,
};

pub struct CNN {
    layers: Vec<Layer>,
}

impl CNN {
    pub fn create_cnn() -> CNN {
        let layers: Vec<Layer> = vec![];

        let cnn: CNN = CNN { layers };

        cnn
    }

    pub fn add_conv_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
    ) {
        let layer: ConvLayer =
            ConvLayer::create_conv_layer(input_size, input_depth, num_filters, kernel_size, stride);
        self.layers.push(Layer::Conv(layer))
    }

    pub fn add_mxpl_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        kernel_size: usize,
        stride: usize,
    ) {
        let layer: MaxPoolingLayer =
            MaxPoolingLayer::create_mxpl_layer(input_size, input_depth, kernel_size, stride);
        self.layers.push(Layer::Mxpl(layer))
    }

    pub fn add_fcl_layer(&mut self, input_width: usize, input_depth: usize, output_size: usize) {
        let layer: FullyConnectedLayer =
            FullyConnectedLayer::create_fcl_layer(input_width, input_depth, output_size);
        self.layers.push(Layer::Fcl(layer))
    }

    pub fn forward_propagate(&mut self, image: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
        let mut output: Vec<Vec<Vec<f32>>> = image;
        for i in 0..self.layers.len() {
            output = self.layers[i].forward_propagate(output);
        }

        output[0][0].clone()
    }

    pub fn last_layer_error(&mut self, label: usize) -> Vec<Vec<Vec<f32>>> {
        let mut error: Vec<f32> = vec![];
        for i in 0..10 {
            let desired: u8 = (label == i) as u8;
            let last_index: usize = self.layers.len() - 1;
            error.push((2.0 / 10.0) * (self.layers[last_index].get_output(i) - desired as f32));
        }

        vec![vec![error.clone()]]
    }

    pub fn back_propagate(&mut self, label: usize) {
        let mut error: Vec<Vec<Vec<f32>>> = self.last_layer_error(label);
        for i in (0..self.layers.len()).rev() {
            error = self.layers[i].back_propagate(error);
        }
    }
}
