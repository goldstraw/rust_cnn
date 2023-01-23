use mnist::{Mnist, MnistBuilder};
use rand::Rng;

use crate::cnn_struct::CNN;

pub fn run() {
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
    cnn.add_conv_layer(28, 1, 6, 5, 1);
    cnn.add_mxpl_layer(24, 6, 2, 2);
    cnn.add_conv_layer(12, 6, 9, 3, 1);
    cnn.add_mxpl_layer(10, 9, 2, 2);
    cnn.add_fcl_layer(5, 9, 10);

    let mut prev: Vec<bool> = vec![false; 100];
    let mut count: u16 = 0;

    while success(&prev) < 0.90 {
        let mut rng = rand::thread_rng();
        let index: usize = rng.gen_range(0..=49_999);

        let output: Vec<f32> = cnn.forward_propagate(vec![train_data[index].clone()]);
        let result: bool = highest_index(output) == train_labels[index];

        cnn.back_propagate(train_labels[index] as usize);

        prev.pop();
        prev.insert(0, result);
        if count % 500 == 499 {
            println!("{}", success(&prev));
        }
        count += 1;
    }
    println!("{}", success(&prev));
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

fn success(prev: &Vec<bool>) -> f32 {
    let mut num_true: u16 = 0;
    for i in 0..prev.len() {
        num_true += prev[i] as u16;
    }

    num_true as f32 / prev.len() as f32
}

fn highest_index(output: Vec<f32>) -> u8 {
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
