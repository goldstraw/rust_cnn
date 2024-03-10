use core::panic;
use std::io::Write;
use std::fs::File;
use std::fmt::{Debug, Formatter};
use std::time::{SystemTime, UNIX_EPOCH};
use std::default::Default;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array3};
use serde::{Serialize, Deserialize};
use crate::activation::Activation;
use crate::util::*;
use crate::optimizer::OptimizerAlg;
// use crate::fiftystates::*;
use crate::mnist::*;

use crate::{
    conv_layer::ConvLayer, dense_layer::DenseLayer, layer::Layer,
    mxpl_layer::MxplLayer,
};

pub struct Hyperparameters {
    pub batch_size: usize,
    pub epochs: usize,
    pub optimizer: OptimizerAlg,
    pub saving_strategy: SavingStrategy,
    pub name: String,
    pub verbose: bool,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Hyperparameters {
            batch_size: 32,
            epochs: 10,
            optimizer: OptimizerAlg::Adam(0.9, 0.999, 1e-8),
            saving_strategy: SavingStrategy::Never,
            name: String::from("model"),
            verbose: true,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct CNN {
    layers: Vec<Layer>,
    layer_order: Vec<String>,
    data: TrainingData,
    minibatch_size: usize,
    creation_time: SystemTime,
    saving_strategy: SavingStrategy,
    training_history: Vec<f32>,
    testing_history: Vec<f32>,
    time_history: Vec<usize>,
    name: String,
    verbose: bool,
    optimizer: OptimizerAlg,
    epochs: usize,
}

impl Debug for CNN {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        let time = self.creation_time.duration_since(UNIX_EPOCH).unwrap().as_millis();
        s.push_str(&format!("File: models/{}_{}.json\n", self.name, time));
        s.push_str(&format!("Time: {}\n", time));
        s.push_str(&format!("Minibatch size: {}\n", self.minibatch_size));
        s.push_str(&format!("Training size: {}\n", self.data.trn_size));
        s.push_str(&format!("Testing size: {}\n", self.data.tst_size));
        s.push_str(&format!("\nLayers:\n"));
        
        for layer in &self.layers {
            s.push_str(&format!("{:?}\n", layer));
        }

        s.push_str(&format!("Training accuracy: {:?}\n", self.training_history));
        s.push_str(&format!("Testing accuracy: {:?}\n", self.testing_history));
        s.push_str(&format!("Time taken: {:?}\n", self.time_history));

        write!(f, "{}", s)
    }
}

impl CNN {
    pub fn new(data: TrainingData, params: Hyperparameters) -> CNN {
        let creation_time = std::time::SystemTime::now();

        let cnn: CNN = CNN {
            layers: vec![],
            layer_order: vec![],
            data,
            minibatch_size: params.batch_size,
            creation_time,
            saving_strategy: params.saving_strategy,
            training_history: vec![],
            testing_history: vec![],
            time_history: vec![],
            name: params.name,
            verbose: params.verbose,
            optimizer: params.optimizer,
            epochs: params.epochs,
        };

        cnn
    }

    pub fn load(model_file_name: &str) -> CNN {
        let model_file = File::open(model_file_name).unwrap();
        let cnn: CNN = serde_json::from_reader(model_file).unwrap();

        cnn
    }

    pub fn add_conv_layer(
        &mut self,
        input_size: (usize, usize, usize),
        num_filters: usize,
        kernel_size: usize,
    ) {
        let conv_layer: ConvLayer = ConvLayer::new(input_size, kernel_size, 1, num_filters, self.optimizer.clone());
        self.layers.push(Layer::Conv(conv_layer));
        self.layer_order.push(String::from("conv"));
    }

    pub fn add_mxpl_layer(
        &mut self,
        input_size: (usize, usize, usize),
        kernel_size: usize,
    ) {
        let mxpl_layer: MxplLayer = MxplLayer::new(input_size, kernel_size, 2);
        self.layers.push(Layer::Mxpl(mxpl_layer));
        self.layer_order.push(String::from("mxpl"));
    }

    pub fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: Activation, dropout: Option<f32>) {
        // Find last layer's output size
        let transition_shape: (usize, usize, usize) = match self.layers.last() {
            Some(Layer::Conv(conv_layer)) => conv_layer.output_size,
            Some(Layer::Mxpl(mxpl_layer)) => mxpl_layer.output_size,
            Some(Layer::Dense(_)) => (input_size, 1, 1),
            None => (input_size, 1, 1),
        };
        let fcl_layer: DenseLayer = DenseLayer::new(input_size, output_size, activation, self.optimizer, dropout, transition_shape);
        self.layers.push(Layer::Dense(fcl_layer));
        self.layer_order.push(String::from("dense"));
    }

    pub fn forward_propagate(&mut self, image: Array3<f32>, training: bool) -> Array1<f32> {
        let mut output: Array3<f32> = image;
        let mut flat_output: Array1<f32> = output.clone().into_shape(output.len()).unwrap();
        for layer in &mut self.layers {
            match layer {
                Layer::Conv(conv_layer) => {
                    output = conv_layer.forward_propagate(output);
                    flat_output = output.clone().into_shape(output.len()).unwrap();
                }
                Layer::Mxpl(mxpl_layer) => {
                    output = mxpl_layer.forward_propagate(output);
                    flat_output = output.clone().into_shape(output.len()).unwrap();
                }
                Layer::Dense(dense_layer) => {
                    flat_output = dense_layer.forward_propagate(flat_output, training);
                }
            }
        }

        flat_output
    }

    pub fn last_layer_error(&mut self, label: usize) -> Array1<f32> {
        let size: usize = match self.layers.last().unwrap() {
            Layer::Dense(dense_layer) => dense_layer.output_size,
            _ => panic!("Last layer is not a DenseLayer"),
        };
        let desired = Array1::<f32>::from_shape_fn(size, |i| (label == i) as usize as f32);
        self.output() - desired
    }

    pub fn back_propagate(&mut self, label: usize, training: bool) {
        let mut flat_error: Array1<f32> = self.last_layer_error(label);
        let mut error: Array3<f32> = flat_error.clone().into_shape((1, 1, flat_error.len())).unwrap();
        for layer in self.layers.iter_mut().rev() {
            match layer {
                Layer::Conv(conv_layer) => {
                    error = conv_layer.back_propagate(error);
                    // flat_error = error.clone().into_shape(error.len()).unwrap();
                }
                Layer::Mxpl(mxpl_layer) => {
                    error = mxpl_layer.back_propagate(error);
                    // flat_error = error.clone().into_shape(error.len()).unwrap();
                }
                Layer::Dense(dense_layer) => {
                    flat_error = dense_layer.back_propagate(flat_error, training);
                    error = flat_error.clone().into_shape(dense_layer.transition_shape).unwrap();
                }
            }
        }
    }

    pub fn update(&mut self, minibatch_size: usize) {
        for layer in &mut self.layers {
            match layer {
                Layer::Conv(conv_layer) => { conv_layer.update(minibatch_size) }
                Layer::Mxpl(_) => { }
                Layer::Dense(dense_layer) => { dense_layer.update(minibatch_size) }
            }
        }
    }

    pub fn output(&self) -> Array1<f32> {
        // self.dense_layers.last().unwrap().output.clone()
        match self.layers.last().unwrap() {
            Layer::Conv(_) => panic!("Last layer is a ConvLayer"),
            Layer::Mxpl(_) => panic!("Last layer is a MxplLayer"),
            Layer::Dense(dense_layer) => dense_layer.output.clone(),
        }
    }

    pub fn get_accuracy(&self, label: usize) -> f32 {
        let mut max = 0.0;
        let mut max_idx = 0;
        let output = self.output();
        for j in 0..output.len() {
            if output[j] > max {
                max = output[j];
                max_idx = j;
            }
        }

        (max_idx == label) as usize as f32
    }

    pub fn train(&mut self) {
        let mut best_train_acc: f32 = self.training_history.last().unwrap_or(&0.0).clone();
        let mut best_test_acc: f32 = self.testing_history.last().unwrap_or(&0.0).clone();
        for epoch in 0..self.epochs {
            let pb = ProgressBar::new((self.data.trn_size / self.minibatch_size) as u64);
            if self.verbose {
                pb.set_style(ProgressStyle::default_bar()
                    .template(&format!("Epoch {}: [{{bar:.cyan/blue}}] {{pos}}/{{len}} - ETA: {{eta}} - acc: {{msg}}", epoch))
                    .unwrap()
                    .progress_chars("#>-"));
            }

            let mut avg_acc = 0.0;
            for i in 0..self.data.trn_size {
                let (image, label) = get_random_image(&self.data);
                let label = *self.data.classes.get(&label).unwrap();
                self.forward_propagate(image, true);
                self.back_propagate(label, true);

                avg_acc += self.get_accuracy(label);

                if i % self.minibatch_size == self.minibatch_size - 1 {
                    self.update(self.minibatch_size);

                    if self.verbose {
                        pb.inc(1);
                        pb.set_message(format!("{:.1}%", avg_acc / (i + 1) as f32 * 100.0));
                    }

                }
                match self.saving_strategy {
                    SavingStrategy::EveryNthEpoch(full_save, n) => {
                        // n is an f32, so save every trn_size / minibatch_size * n iterations
                        let every_n = (self.data.trn_size as f32 * n) as usize;
                        if i % every_n == every_n - 1 {
                            self.save(full_save);
                        }
                    }
                    _ => {}
                }

            }
            
            avg_acc /= self.data.trn_size as f32;
            if self.verbose {
                pb.set_message(format!("{:.1}% - Testing...", avg_acc));
            }
            
            // Testing
            let mut avg_test_acc = 0.0;
            for _i in 0..self.data.tst_size {
                // let image: Array3<f32> = self.data.tst_img[i].clone();
                let (image, label) = get_random_test_image(&self.data);
                let label = *self.data.classes.get(&label).unwrap();
                self.forward_propagate(image, false);
                
                avg_test_acc += self.get_accuracy(label);//self.data.tst_lbl[i] as usize);
            }
            
            avg_test_acc /= self.data.tst_size as f32;
            if self.verbose {
                pb.finish_with_message(format!("{:.1}% - Test: {:.1}%", avg_acc * 100.0, avg_test_acc * 100.0));
            }

            self.training_history.push(avg_acc);
            self.testing_history.push(avg_test_acc);
            let duration = SystemTime::now().duration_since(self.creation_time).unwrap();
            self.time_history.push(duration.as_secs() as usize);
            
            match self.saving_strategy {
                SavingStrategy::EveryEpoch(full_save) => {
                    self.save(full_save);
                }
                SavingStrategy::BestTrainingAccuracy(full_save) => {
                    if avg_acc > best_train_acc {
                        best_train_acc = avg_acc;
                        self.save(full_save);
                    } else {
                        // If the accuracy is not improving, save the metadata anyway
                        self.save(false);
                    }
                }
                SavingStrategy::BestTestingAccuracy(full_save) => {
                    if avg_test_acc > best_test_acc {
                        best_test_acc = avg_test_acc;
                        self.save(full_save);
                    } else {
                        // If the accuracy is not improving, save the metadata anyway
                        self.save(false);
                    }
                }
                _ => {}
            }
        }
    }

    pub fn zero(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Layer::Conv(conv_layer) => { conv_layer.zero() }
                Layer::Mxpl(mxpl_layer) => { mxpl_layer.zero() }
                Layer::Dense(dense_layer) => { dense_layer.zero() }
            }
        }
    }

    pub fn save(&self, full_save: bool) {
        std::fs::create_dir_all("models").unwrap();
        let time_str = self.creation_time.duration_since(UNIX_EPOCH).unwrap().as_millis();
        if full_save {
            let model_file_name = format!("models/{}_{}.json", self.name, time_str);
            let model_file = std::fs::File::create(&model_file_name).unwrap();
            serde_json::to_writer(model_file, &self).unwrap();
        }

        // Write metadata to models/model_{}.txt
        let metadata_file_name = format!("models/{}_{}.txt", self.name, time_str);
        let mut metadata_file = std::fs::File::create(&metadata_file_name).unwrap();
        write!(metadata_file, "{:?}", self).unwrap();
    }

    // Top N accuracy with only one model
    pub fn top_n_accuracy(&mut self, data: TrainingData) -> (Vec<usize>, usize) {
        let mut corrects: Vec<usize> = vec![0; 10];
        let mut total: usize = 0;

        for i in 0..data.tst_size {
            println!("{} / {} | Top {} Accuracy: {}/{} = {}", i, data.tst_size, (i%10) + 1, corrects[i%10], total, corrects[i%10] as f64 / total as f64);
            let image = match data.tst_img[i].clone() {
                TrainImage::Path(p) => load_image(&p).unwrap(),
                TrainImage::Image(a) => a,
            };
            let state_name = idx_to_state(data.tst_lbl[i]);
            let output = self.forward_propagate(image.clone(), false);

            // Sort the state output
            let mut state_output_vec: Vec<(usize, f32)> = output.iter().enumerate().map(|(i, v)| (i, *v)).collect();
            state_output_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Check if the correct state is in the top n
            // for i in 0..10 {
            for i in 0..state_output_vec.len() {
                if state_output_vec[i].0 == state_to_idx(state_name) {
                    for j in i..10 {
                        corrects[j] += 1;
                    }
                    break;
                }
            }

            total += 1;
        }

        (corrects, total)
    }
}