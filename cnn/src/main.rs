use rand::Rng;
use rand_distr::{Distribution, Normal};
use mnist::*;

const LEARNING_RATE: f32 = 0.01;

struct Kernel {
    size: usize,
    depth: usize,
    val: Vec<Vec<Vec<f32>>>,
}

impl Kernel {
    fn create_kernel(size: usize, depth: usize) -> Kernel {
        let mut val: Vec<Vec<Vec<f32>>> = vec![];
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
    biases: Vec<f32>,
    output: Vec<Vec<f32>>,
}

impl ConvLayer {
    fn create_conv_layer(input_size: usize, input_depth: usize, num_filters: usize, kernel_size: usize, stride: usize) -> ConvLayer {

        let mut biases: Vec<f32> = vec![];
        let mut kernels: Vec<Kernel> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for _ in 0..num_filters {
            biases.push(normal.sample(&mut rand::thread_rng()));
            kernels.push(Kernel::create_kernel(kernel_size, input_depth));
        }

        let output_size: usize = ((input_size-kernel_size) / stride) + 1;
        let mut output: Vec<Vec<f32>> = vec![];
        for h in 0..output_size {
            output.push(vec![]);
            for _ in 0..output_size {
                output[h].push(0.0);
            }
        }
        
        let layer: ConvLayer = ConvLayer {
            input_size,
            input_depth,
            num_filters,
            kernel_size,
            output_size,
            stride,
            biases,
            output,
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
    output: Vec<Vec<f32>>,
}

impl MaxPoolingLayer {
    fn create_mxpl_layer(input_size: usize, input_depth: usize, kernel_size: usize, stride: usize) -> MaxPoolingLayer {
        let output_size: usize = ((input_size-kernel_size) / stride) + 1;
        let mut output: Vec<Vec<f32>> = vec![];
        for h in 0..output_size {
            output.push(vec![]);
            for _ in 0..output_size {
                output[h].push(0.0);
            }
        }

        let layer: MaxPoolingLayer = MaxPoolingLayer {
            input_size,
            input_depth,
            kernel_size,
            output_size,
            stride,
            output,
        };

        layer
    }
}

struct FullyConnectedLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    output: Vec<f32>,
}

impl FullyConnectedLayer {
    fn create_fcl_layer(input_size: usize, output_size: usize) -> FullyConnectedLayer {
        
        let mut biases: Vec<f32> = vec![];
        let mut weights: Vec<Vec<f32>> = vec![];
        let normal = Normal::new(0.0, 0.1).unwrap();
        for i in 0..input_size {
            biases.push(normal.sample(&mut rand::thread_rng()));
            weights.push(vec![]);
            for _ in 0..output_size {
                weights[i].push(normal.sample(&mut rand::thread_rng()));
            }
        }

        let mut output: Vec<f32> = vec![];
        for _ in 0..output_size {
            output.push(0.0);
        }

        let layer: FullyConnectedLayer = FullyConnectedLayer {
            input_size,
            output_size,
            weights,
            biases,
            output,
        };

        layer
    }

    fn forward_propagate(&mut self, input: &Vec<f32>) {
        for j in 0..self.output_size {
            self.output[j] = self.biases[j];
            for i in 0..self.input_size {
                self.output[j] += input[i] * self.weights[i][j];
            }
            self.output[j] = sigmoid(self.output[j]);
        }
    }
}

enum Layer {
    Conv(ConvLayer),
    Mxpl(MaxPoolingLayer),
    Fcl(FullyConnectedLayer),
}

impl Layer {
    fn forward_propagate(&mut self, input: &Vec<Vec<f32>>) {
        match self {
            //Layer::Conv(a) => a.forward_propagate(input),
            Layer::Conv(_) => panic!("Not defined yet"),
            // Layer::Mxpl(b) => b.forward_propagate(input),
            Layer::Mxpl(_) => panic!("Not defined yet"),
            Layer::Fcl(c) => c.forward_propagate(&flatten(input)),
        }
    }
}

struct CNN {
    layers: Vec<Layer>,
}

impl CNN {
    fn create_cnn() -> CNN {
        let mut layers: Vec<Layer> = vec![];
        
        let cnn: CNN = CNN {
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

    fn add_fcl_layer(&mut self, input_size: usize, output_size: usize) {
        let mut layer: FullyConnectedLayer = FullyConnectedLayer::create_fcl_layer(input_size, output_size);
        self.layers.push(Layer::Fcl(layer))
    }

    fn forward_propagate(&mut self, image: &Vec<Vec<f32>>) -> &Vec<f32> {

        for i in 0..self.layers.len() {
            self.layers[i].forward_propagate(image);
        }

        match &self.layers[self.layers.len()-1] {
            Layer::Fcl(FullyConnectedLayer {output, ..}) => &output,
            _ => panic!("The last layer is not a FullyConnectedLayer"),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn format_images(data: Vec<u8>, num_images: usize) -> Vec<Vec<Vec<f32>>> {
    let img_width: usize = 28;
    let img_height: usize = 28;

    let mut images: Vec<Vec<Vec<f32>>> = vec![];
    for image_count in 0..num_images {
        let mut image: Vec<Vec<f32>> = vec![];
        for h in 0..img_height {
            let mut row: Vec<f32> = vec![];
            for w in 0..img_width {
                let i: usize = (image_count * 28 * 28) + (h * 28) + w;
                row.push(data[i] as f32 / 256.0);
            }
            image.push(row);
        }
        images.push(image);
    }

    images
}

fn flatten(square: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut flat_data: Vec<f32> = vec![];
    for row in square {
        flat_data.extend(row);
    }

    flat_data
}

fn success(prev: &Vec<bool>) -> f32 {
    let mut num_true: u16 = 0;
    for i in 0..prev.len() {
        num_true += prev[i] as u16;
    }

    num_true as f32 / prev.len() as f32
}

fn highest_index(output: &Vec<f32>) -> u8 {
    let mut highest_index: u8 = 127;
    let mut highest_value: f32 = 0.0;

    for i in 0..output.len() {
        if output[i] > highest_value {
            highest_value = output[i];
            highest_index = i as u8;
        }
    }

    return highest_index;
}

fn main() {

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data: Vec<Vec<Vec<f32>>> = format_images(trn_img, 50_000);
    let train_labels: Vec<u8> = trn_lbl;

    let _test_data: Vec<Vec<Vec<f32>>> = format_images(tst_img, 10_000);
    let _test_labels: Vec<u8> = tst_lbl;

    let mut cnn: CNN = CNN::create_cnn();
    cnn.add_conv_layer(28, 3, 6, 5, 1);
    cnn.add_mxpl_layer(24, 6, 2, 2);
    cnn.add_conv_layer(12, 6, 9, 3, 1);
    cnn.add_mxpl_layer(10, 6, 2, 2);
    cnn.add_fcl_layer(225, 10);

    let mut prev: Vec<bool> = vec![false; 100];

    while success(&prev) < 0.95 {
        let mut rng = rand::thread_rng();
        let index: usize = rng.gen_range(0..=49_999);
        let output: &Vec<f32> = cnn.forward_propagate(&train_data[index]);
        let result: bool = highest_index(output) == train_labels[index];

        // cnn.back_propagate(train_labels[index]);

        prev.pop();
        prev.insert(0, result);

        if rng.gen_range(0..=500) == 0 {
            println!("{}", success(&prev));
        }
    }

    // # CNN
    // # Input                 [28x28x1]
    // # ConvLayer (5x5),1     [24x24x6]
    // # MaxPool   (2x2),2     [12x12x6]
    // # ConvLayer (3x3),1     [10x10x9]
    // # MaxPool   (2x2),2     [5x5x9]
    // # FullLayer (225x10)    [225]
}