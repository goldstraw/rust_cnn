use rand_distr::{Distribution, Normal};

struct ConvLayer {
    input_size: [usize; 3],
    num_filters: usize,
    kernel_size: [usize; 2],
    output_size: [usize; 2],
    stride: usize,
    biases: Vec<f64>,
    learning_rate: f64,
}

impl ConvLayer {
    fn create_conv_layer(input_size: [usize; 3], num_filters: usize, kernel_size: [usize; 2], stride: usize) -> ConvLayer {

        // self.kernels = [[[[random.gauss(0,0.1) for j in range(kernel_size)] for i in range(kernel_size)] for k in range(input_depth)] for l in range(num_filters)]

        let mut biases: Vec<f64> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for _ in 0..num_filters {
            biases.push(normal.sample(&mut rand::thread_rng()));
        }
        
        let layer: ConvLayer = ConvLayer {
            input_size,
            num_filters,
            kernel_size,
            output_size: [((input_size[0]-kernel_size[0]) / stride) + 1; 2],
            stride,
            biases,
            learning_rate: 0.01,
        };

        layer
    }
}

enum Layer {
    Conv(ConvLayer),
}

struct CNN {
    num_layers: usize,
    layers: Vec<Layer>,
}

impl CNN {
    fn create_cnn() -> CNN {
        let mut layers: Vec<Layer> = vec![];
        
        let cnn: CNN = CNN {
            num_layers: 0,
            layers,
        };
        
        cnn
    }
    
    fn add_conv_layer(&mut self, input_size: [usize; 3], num_filters: usize, kernel_size: [usize; 2], stride: usize) {
        let mut layer: ConvLayer = ConvLayer::create_conv_layer(input_size, num_filters, kernel_size, stride);
        self.layers.push(Layer::Conv(layer))
    }
}

fn main() {

    let mut cnn: CNN = CNN::create_cnn();
    cnn.add_conv_layer([28, 28, 3], 6, [5, 5], 1);
}