use crate::{
    conv_layer::ConvLayer, fully_connected_layer::FullyConnectedLayer, layer::Layer,
    max_pooling_layer::MaxPoolingLayer,
};

/// A struct that represents a Convolutional Neural Network (CNN)
pub struct CNN {
    /// A vector of `Layer` objects representing the layers in the CNN
    layers: Vec<Layer>,
}

impl CNN {
    /// Creates a new `CNN` object with an empty vector of layers
    pub fn new() -> CNN {
        let layers: Vec<Layer> = vec![];

        let cnn: CNN = CNN { layers };

        cnn
    }

    /// Adds a convolutional layer to the neural network.
    pub fn add_conv_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
    ) {
        // Create a new convolutional layer with the specified parameters.
        let layer: ConvLayer =
            ConvLayer::new(input_size, input_depth, num_filters, kernel_size, stride);
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(Layer::Conv(layer))
    }

    /// Adds a max pooling layer to the neural network
    pub fn add_mxpl_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        kernel_size: usize,
        stride: usize,
    ) {
        // Create a new max pooling layer with the specified parameters
        let layer: MaxPoolingLayer =
            MaxPoolingLayer::new(input_size, input_depth, kernel_size, stride);
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(Layer::Mxpl(layer))
    }

    /// Adds a fully connected layer to the neural network
    pub fn add_fcl_layer(&mut self, input_width: usize, input_depth: usize, output_size: usize) {
        // Create a new fully connected layer with the specified parameters
        let layer: FullyConnectedLayer =
            FullyConnectedLayer::new(input_width, input_depth, output_size);
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(Layer::Fcl(layer))
    }

    /// Forward propagates an input matrix through the CNN.
    pub fn forward_propagate(&mut self, image: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
        let mut output: Vec<Vec<Vec<f32>>> = image;

        // Forward propagate through each layer of the network
        for i in 0..self.layers.len() {
            output = self.layers[i].forward_propagate(output);
        }

        // Flatten and return the output of the final layer
        output[0][0].clone()
    }

    /// Calculates the error of the last layer of the network
    pub fn last_layer_error(&mut self, label: usize) -> Vec<Vec<Vec<f32>>> {
        let mut error: Vec<f32> = vec![];
        // Calculate the error for each output neuron
        for i in 0..10 {
            let desired: u8 = (label == i) as u8;
            let last_index: usize = self.layers.len() - 1;
            error.push((2.0 / 10.0) * (self.layers[last_index].get_output(i) - desired as f32));
        }

        vec![vec![error.clone()]]
    }

    /// Backpropagate the error from the output layer to the input layer
    pub fn back_propagate(&mut self, label: usize) {
        // Retrieve the last layer error to backpropagate
        let mut error: Vec<Vec<Vec<f32>>> = self.last_layer_error(label);
        
        // Iterate backwards through the layers and backpropagate the error
        for i in (0..self.layers.len()).rev() {
            error = self.layers[i].back_propagate(error);
        }
    }
}
