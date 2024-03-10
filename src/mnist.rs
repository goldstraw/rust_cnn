use std::path::Path;
use rust_mnist::Mnist;
use crate::util::{TrainingData, TrainImage, load_image};
use std::collections::HashMap;
use ndarray::Array3;
use rand::seq::IteratorRandom;

pub fn load_mnist<T>(mnist_path: T) -> TrainingData
where T: AsRef<Path>
{
    let (rows, cols) = (28, 28);
    let mnist_path = mnist_path.as_ref();
    let mnist = Mnist::new(mnist_path.to_str().unwrap());

    let mut trn_img = Vec::<TrainImage>::new();
    let mut trn_lbl = Vec::<usize>::new();
    let mut tst_img = Vec::<TrainImage>::new();
    let mut tst_lbl = Vec::<usize>::new();

    // Make unpacked folder inside mnist_path
    let mnist_path = mnist_path.join("unpacked");
    if !mnist_path.exists() {
        std::fs::create_dir(mnist_path.as_path()).expect("Failed to create unpacked folder.");
    }

    for i in 0..60000 {
        let mut img: Array3<f32> = Array3::<f32>::zeros((rows, cols, 1));

        for j in 0..rows {
            for k in 0..cols {
                img[[j, k, 0]] = mnist.train_data[i][(j * 28 + k) as usize] as f32 / 255.0;
            }
        }
        trn_img.push(TrainImage::Image(img));
        trn_lbl.push(mnist.train_labels[i] as usize);
    }

    for i in 0..10000 {
        let mut img: Array3<f32> = Array3::<f32>::zeros((rows, cols, 1));

        for j in 0..rows {
            for k in 0..cols {
                img[[j, k, 0]] = mnist.test_data[i][(j * 28 + k) as usize] as f32 / 255.0;
            }
        }
        tst_img.push(TrainImage::Image(img));
        tst_lbl.push(mnist.test_labels[i] as usize);
    }

    // 'classes' allows us to only train on a subset of the data
    // Here, we use all 10 classes

    let classes: HashMap<usize, usize> = (0..10).enumerate().collect();

    let training_data: TrainingData = TrainingData {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        rows,
        cols,
        trn_size: 60000,
        tst_size: 10000,
        classes,
    };

    training_data
}



pub fn get_random_image(data: &TrainingData) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = data.trn_img.iter().zip(data.trn_lbl.iter()).choose(&mut rng).unwrap();
    match img {
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_image(img_path).unwrap();
            (img, *label)
        }
    }
}

pub fn get_random_test_image(data: &TrainingData) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = data.tst_img.iter().zip(data.tst_lbl.iter()).choose(&mut rng).unwrap();
    match img {
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_image(img_path).unwrap();
            (img, *label)
        }
    }
}