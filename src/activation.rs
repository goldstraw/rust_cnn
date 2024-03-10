use ndarray::Array1;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Activation {
    Relu,
    Sigmoid,
    Softmax,
}

// How to make this generic?
pub fn forward(x: Array1<f32>, activation: Activation) -> Array1<f32> {
    match activation {
        Activation::Relu => relu(x),
        Activation::Sigmoid => sigmoid(x),
        Activation::Softmax => softmax(x),
    }
}

pub fn backward(x: Array1<f32>, activation: Activation) -> Array1<f32> {
    match activation {
        Activation::Relu => relu_derivative(x),
        Activation::Sigmoid => sigmoid_derivative(x),
        Activation::Softmax => softmax_derivative(x),
    }
}

fn softmax(x: Array1<f32>) -> Array1<f32> {
    let max = x.fold(x[0], |acc, &xi| if xi > acc { xi } else { acc });
    let exps = x.mapv(|xi| (xi - max).exp());
    let sum: f32 = exps.sum();
    exps / sum
}

fn softmax_derivative(x: Array1<f32>) -> Array1<f32> {
    Array1::ones(x.len())
}

fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|xi| 1.0 / (1.0 + (-xi).exp()))
}

fn sigmoid_derivative(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|xi| xi * (1.0 - xi))
}

fn relu(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|xi| if xi > 0.0 { xi } else { 0.0 })
}

fn relu_derivative(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|xi| if xi > 0.0 { 1.0 } else { 0.0 })
}