use oxi_net::cnn::*;
// use oxi_net::fiftystates::load_50states10k;
use oxi_net::mnist::load_mnist;
use oxi_net::optimizer::OptimizerAlg;
use oxi_net::activation::Activation;


fn main() {
    // Load MNIST dataset
    let data = load_mnist("./data/");

    // Set hyperparameters
    let hyperparameters = Hyperparameters {
        batch_size: 10,
        epochs: 10,
        optimizer: OptimizerAlg::SGD(0.1),
        ..Hyperparameters::default()
    };

    // Create CNN architecture
    let mut cnn = CNN::new(data, hyperparameters);
    cnn.set_input_shape(vec![28, 28, 3]);
    cnn.add_conv_layer(8, 3);
    cnn.add_mxpl_layer(2);
    cnn.add_dense_layer(128, Activation::Relu, Some(0.25));
    cnn.add_dense_layer(64, Activation::Relu, Some(0.25));
    cnn.add_dense_layer(10, Activation::Softmax, None);

    cnn.train();

}

// Example CNN for 50States10K dataset
// To run this example, download the 50States10K dataset 
// to the root of the project
/*
fn main() {
    // Filter for only 2 states for faster training with 95% train/test split.
    // Note: to save on memory, the images are loaded on demand, which slows
    // down training. Smaller datasets, loaded into memory, will train faster.
    let filter = Some(vec!["Hawaii", "Alaska"]);
    let data = load_50states10k("../oxi_net/50States10K/", 0.95, filter.clone()).unwrap();

    // Set hyperparameters
    let hyperparameters = Hyperparameters {
        batch_size: 10,
        epochs: 10,
        optimizer: OptimizerAlg::RMSProp(0.001, 0.9),
        ..Hyperparameters::default()
    };

    // Create CNN architecture
    let mut cnn = CNN::new(data, hyperparameters);
    cnn.add_mxpl_layer((256, 256, 3), 2);
    cnn.add_mxpl_layer((128, 128, 3), 2);
    cnn.add_mxpl_layer((64, 64, 3), 2);
    cnn.add_conv_layer((32, 32, 3), 8, 3);
    cnn.add_mxpl_layer((30, 30, 8), 2);
    cnn.add_dense_layer(15 * 15 * 8, 256, Activation::Relu, Some(0.25));
    cnn.add_dense_layer(256, 128, Activation::Relu, Some(0.25));
    let output_neurons = match filter {
        Some(f) => f.len(),
        None => 50,
    };
    cnn.add_dense_layer(128, output_neurons, Activation::Softmax, None);

    cnn.train();
}
*/
