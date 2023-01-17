use rand::random;

struct Neuron {
    activation: f64,
    bias: f64,
    weights: Vec<f64>,
    error: f64,
}

struct MLP {
    layer_sizes: Vec<usize>,
    num_layers: usize,
    layers: Vec<Vec<Neuron>>,
    learning_rate: f64,
}

impl MLP {
    fn build_mlp(layer_sizes: Vec<usize>, learning_rate: f64) -> MLP {
        let num_layers: usize = layer_sizes.len();
        let mut layers: Vec<Vec<Neuron>> = vec![];
        for i in 0..num_layers {
            layers.push(vec![]);
            for _ in 0..layer_sizes[i] {
                let next_layer_size: usize = if i < (num_layers-1) { layer_sizes[i+1] } else { 0 };
                let mut weights: Vec<f64> = vec![];
                for _ in 0..next_layer_size {
                    weights.push(random::<f64>());
                }
                let neuron: Neuron = Neuron {
                    activation: random::<f64>(),
                    bias: random::<f64>(),
                    weights,
                    error: 0.0,
                };
                layers[i].push(neuron);
            }
        }
        let mlp: MLP = MLP {
            layer_sizes,
            num_layers,
            layers,
            learning_rate,
        };
        
        mlp
    }
    
    fn forward_propagate(&mut self, input_layer: Vec<f64>) {
        for i in 0..self.layer_sizes[0] {
            self.layers[0][i].activation = input_layer[i];
        }
        
        for i in 1..self.num_layers {
            for j in 0..self.layer_sizes[i] {
                self.layers[i][j].activation = 0.0;
                for k in 0..self.layer_sizes[i-1] {
                    self.layers[i][j].activation += self.layers[i-1][k].activation * self.layers[i-1][k].weights[j];
                }
                self.layers[i][j].activation = if self.layers[i][j].activation > self.layers[i][j].bias { 1.0 } else { 0.0 };
            }
        }
    }

    fn back_propagate(&mut self, desired: &Vec<f64>) {
        let n: f64 = self.layer_sizes[self.num_layers-1] as f64;
        for i in 0..n as usize {
            let output: &mut Neuron = &mut self.layers[self.num_layers-1][i];
            let error: f64 = (2.0/n) * (desired[i]-output.activation) * self.learning_rate;
            output.error = error;
            output.bias -= error;
        }
        
        for i in 0..self.num_layers-1 {
            let layer: usize = self.num_layers - (2+i);
            for j in 0..self.layer_sizes[layer] {
                self.layers[layer][j].error = 0.0;
                for k in 0..self.layer_sizes[layer+1] {
                    let next_error: f64 = self.layers[layer+1][k].error;
                    self.layers[layer][j].error += self.layers[layer][j].weights[k] * next_error;
                    self.layers[layer][j].weights[k] += self.layers[layer][j].activation * next_error;
                }
                self.layers[layer][j].bias -= self.layers[layer][j].error;
            }
        }
    }
    
    fn is_correct(&mut self, desired: &Vec<f64>) -> bool {
        for i in 0..4 {
            if self.layers[self.num_layers-1][i].activation != desired[i] {
                return false;
            }
        }
        true
    }
}

fn main() {
    let data: [(f64, f64, u8); 48] = [
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

    let layer_sizes: Vec<usize> = vec![2, 15, 4];
    let mut mlp: MLP = MLP::build_mlp(layer_sizes, 0.02);
    let mut success: f64 = 0.0;
    
    while success < 0.95 {
        success = 0.0;
        for sample in data {
            mlp.forward_propagate([sample.0,sample.1].to_vec());
            let mut desired: Vec<f64> = vec![0.0; mlp.layer_sizes[mlp.num_layers-1]];
            desired[sample.2 as usize] = 1.0;
            mlp.back_propagate(&desired);
            if mlp.is_correct(&desired) {
                success += 1.0;
            }
        }
        success /= data.len() as f64;
        println!("{}", success);
    }
}