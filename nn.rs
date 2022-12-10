use rand::random;

struct Neuron {
    activation: f32,
    bias: f32,
    weights: Vec<f32>,
    error: f32,
}

struct MLP {
    layer_sizes: Vec<u16>,
    num_layers: u8,
    layers: Vec<Vec<Neuron>>,
    learning_rate: f32,
}

fn build_mlp(layer_sizes: Vec<u16>, learning_rate: f32) -> MLP {
    let num_layers: u8 = layer_sizes.len() as u8;
    let mut layers: Vec<Vec<Neuron>> = vec![];
    for i in 0..num_layers {
        layers.push(vec![]);
        for j in 0..layer_sizes[i] {
            let neuron: Neuron = Neuron {
                activation: random::<f64>(),
                bias: random::<f64>(),
                weights: ,
                use rand::random;error: 
                
                0,
            }
            layers.get(i).push(neuron);
        }
    }
    let mlp: MLP = MLP {
        layer_sizes,
        num_layers,
        layers,
        learning_rate,
    };
    println!("hi");
    
    mlp
}

fn main() {
    let data: [(f32, f32, u8); 48] = [
        (0.56, 0.93, 0), (0.72, 0.96, 0), (0.90, 0.95, 0), (0.69, 0.81, 0),
        (0.83, 0.84, 0), (0.75, 0.66, 0), (0.84, 0.69, 0), (0.96, 0.83, 0),
        (0.62, 0.73, 0), (0.95, 0.63, 0), (0.84, 0.58, 0), (0.68, 0.59, 0), 
        (0.04, 0.45, 1), (0.40, 0.60, 1), (0.22, 0.54, 1), (0.37, 0.50, 1),
        (0.55, 0.50, 1), (0.21, 0.44, 1), (0.10, 0.35, 1), (0.49, 0.32, 1),
        (0.05, 0.15, 1), (0.33, 0.34, 1), (0.19, 0.25, 1), (0.39, 0.19, 1),
        (0.07, 0.03, 1), (0.24, 0.09, 1), (0.44, 0.03, 1), (0.02, 0.79, 2),
        (0.16, 0.78, 2), (0.29, 0.72, 2), (0.46, 0.75, 2), (0.21, 0.96, 2),
        (0.07, 0.91, 2), (0.36, 0.86, 2), (0.05, 0.62, 2), (0.18, 0.64, 2),
        (0.93, 0.47, 3), (0.96, 0.40, 3), (0.70, 0.48, 3), (0.83, 0.47, 3),
        (0.65, 0.33, 3), (0.74, 0.40, 3), (0.86, 0.33, 3), (0.61, 0.20, 3),
        (0.87, 0.24, 3), (0.67, 0.08, 3), (0.87, 0.06, 3), (0.93, 0.01, 3)
    ];

    loop {
        for sample in data {
            println!("{:?}", sample);
        }
        let layer_sizes: Vec<u16> = vec![2, 15, 4];
        build_MLP(layer_sizes, 0.02);
        break;
    }
}