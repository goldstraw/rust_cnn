/// A trait for a layer in a CNN
pub trait Layer {
    /// Forward propagates input through the layer
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>;

    /// Back propagates error through the layer
    fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>;

    /// Returns the output value at a specific index
    fn get_output(&mut self, index: usize) -> f32;
}