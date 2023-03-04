use crate::conv_layer::ConvLayer;
use crate::max_pooling_layer::MaxPoolingLayer;
use crate::fully_connected_layer::FullyConnectedLayer;

/// Flattens a 3D vector into a 1D vector.
fn flatten(squares: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let mut flat_data: Vec<f32> = vec![];

    for square in squares {
        for row in square {
            flat_data.extend(row);
        }
    }

    flat_data
}

/// Represents a layer in a neural network.
pub enum Layer {
    Conv(ConvLayer),
    Mxpl(MaxPoolingLayer),
    Fcl(FullyConnectedLayer),
}

impl Layer {
    /// Forward propagates input through the layer
    pub fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        match self {
            Layer::Conv(a) => a.forward_propagate(input),
            Layer::Mxpl(b) => b.forward_propagate(input),
            Layer::Fcl(c) => c.forward_propagate(flatten(input)),
        }
    }

    /// Back propagates error through the layer
    pub fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        match self {
            Layer::Conv(a) => a.back_propagate(error),
            Layer::Mxpl(b) => b.back_propagate(error),
            Layer::Fcl(c) => c.back_propagate(error[0][0].clone()),
        }
    }

    /// Returns the output value at a specific index
    pub fn get_output(&mut self, index: usize) -> f32 {
        match self {
            Layer::Conv(_) => panic!("Layer not fully connected"),
            Layer::Mxpl(_) => panic!("Layer not fully connected"),
            Layer::Fcl(c) => c.output[index],
        }
    }
}