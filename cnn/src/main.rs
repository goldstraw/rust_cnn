use rand;
use rand_distr::{Distribution, Normal};

struct Kernel {
    size: usize,
    depth: usize,
    val: Vec<Vec<Vec<f64>>>,
}

impl Kernel {
    fn create_kernel(size: usize, depth: usize) -> Kernel {
        let mut val: Vec<Vec<Vec<f64>>> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for i in 0..depth {
            val.push(vec![]);    
            for j in 0..size {
                val[i].push(vec![]);
                for _ in 0..size {
                    val[i][j].push(normal.sample(&mut rand::thread_rng()));
                }
            }
        }
        
        let kernel: Kernel = Kernel {
            size,
            depth,
            val,
        };

        kernel
    }
}

struct ConvLayer {
    input_size: usize,
    input_depth: usize,
    num_filters: usize,
    kernel_size: usize,
    output_size: usize,
    stride: usize,
    biases: Vec<f64>,
    learning_rate: f64,
}

impl ConvLayer {
    fn create_conv_layer(input_size: usize, input_depth: usize, num_filters: usize, kernel_size: usize, stride: usize) -> ConvLayer {

        let mut biases: Vec<f64> = vec![];
        let mut kernels: Vec<Kernel> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for _ in 0..num_filters {
            biases.push(normal.sample(&mut rand::thread_rng()));
            kernels.push(Kernel::create_kernel(kernel_size, input_depth));
        }
        
        let layer: ConvLayer = ConvLayer {
            input_size,
            input_depth,
            num_filters,
            kernel_size,
            output_size: ((input_size-kernel_size) / stride) + 1,
            stride,
            biases,
            learning_rate: 0.01,
        };

        layer
    }
}

struct MaxPoolingLayer {
    input_size: usize,
    input_depth: usize,
    kernel_size: usize,
    output_size: usize,
    stride: usize,
}

impl MaxPoolingLayer {
    fn create_mxpl_layer(input_size: usize, input_depth: usize, kernel_size: usize, stride: usize) -> MaxPoolingLayer {
        let layer: MaxPoolingLayer = MaxPoolingLayer {
            input_size,
            input_depth,
            kernel_size,
            output_size: ((input_size-kernel_size) / stride) + 1,
            stride,
        };

        layer
    }
}

enum Layer {
    Conv(ConvLayer),
    Mxpl(MaxPoolingLayer),
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
    
    fn add_conv_layer(&mut self, input_size: usize, input_depth: usize, num_filters: usize, kernel_size: usize, stride: usize) {
        let mut layer: ConvLayer = ConvLayer::create_conv_layer(input_size, input_depth, num_filters, kernel_size, stride);
        self.layers.push(Layer::Conv(layer))
    }

    fn add_mxpl_layer(&mut self, input_size: usize, input_depth: usize, kernel_size: usize, stride: usize) {
        let mut layer: MaxPoolingLayer = MaxPoolingLayer::create_mxpl_layer(input_size, input_depth, kernel_size, stride);
        self.layers.push(Layer::Mxpl(layer))
    }
}

fn main() {

    let mut cnn: CNN = CNN::create_cnn();
    cnn.add_conv_layer(28, 3, 6, 5, 1);
    cnn.add_mxpl_layer(24, 6, 2, 2);
    cnn.add_conv_layer(12, 6, 9, 3, 1);
    cnn.add_mxpl_layer(10, 6, 2, 2);


    // # CNN
    // # Input                 [28x28x1]
    // # ConvLayer (5x5),1     [24x24x6]
    // # MaxPool   (2x2),2     [12x12x6]
    // # ConvLayer (3x3),1     [10x10x9]
    // # MaxPool   (2x2),2     [5x5x9]
    // # FullLayer (25x10)     [225]
}