use serde::{Serialize, Deserialize};
use std::fmt::{Debug, Formatter};
use crate::conv_layer::ConvLayer;
use crate::mxpl_layer::MxplLayer;
use crate::dense_layer::DenseLayer;

#[derive(Serialize, Deserialize)]
pub enum Layer {
    Conv(ConvLayer),
    Mxpl(MxplLayer),
    Dense(DenseLayer),
}

impl Debug for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Conv(layer) => write!(f, "{:?}", layer),
            Layer::Mxpl(layer) => write!(f, "{:?}", layer),
            Layer::Dense(layer) => write!(f, "{:?}", layer),
        }
    }
}