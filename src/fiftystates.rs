use crate::util::{TrainingData, idx_to_state, load_image};
use image::io::Reader as ImageReader;
use ndarray::Array3;
use std::path::Path;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use walkdir::WalkDir;
use rand::seq::IteratorRandom;

pub fn load_50states10k<T>(base_path: T, train_prob: f64, filter: Option<Vec<&str>>) -> Result<TrainingData, String>
where T: AsRef<Path>
{
    let mut trn_img = Vec::new();
    let mut trn_lbl = Vec::new();
    let mut tst_img = Vec::new();
    let mut tst_lbl = Vec::new();
    let rows = 256;
    let cols = 256;
    let train_cutoff = (train_prob * u64::MAX as f64) as u64;
    let mut classes: HashMap<usize, usize> = HashMap::new();
    
    for i in 0..50 {
        if let Some(filter) = &filter {
            if !filter.contains(&idx_to_state(i)) {
                continue;
            }
        }
        classes.insert(i, classes.len());
        let state = idx_to_state(i);
        let folder_path = base_path.as_ref().join(state);
        if !folder_path.exists() {
            return Err(format!("Folder {} does not exist.", folder_path.to_str().unwrap()));
        }
        
        for entry in WalkDir::new(folder_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file() && e.path().extension().map(|s| s == "jpg").unwrap_or(false))
        {
            let mut base_file_name = entry.path().file_name().unwrap().to_str().unwrap().to_string();
            // Remove rotation from file name, so that the same location with different rotations is not
            // both in training and testing
            let rotations = vec!["_0.jpg", "_90.jpg", "_180.jpg", "_270.jpg"];
            for rotation in rotations {
                base_file_name = (&base_file_name).replace(rotation, "");
            }
            // Hash base_file_name to decide whether to put in training or testing
            let hash: u64 = {
                let mut hasher = DefaultHasher::new();
                base_file_name.hash(&mut hasher);
                hasher.finish()
            };

            if hash < train_cutoff {
                trn_img.push(entry.path().to_path_buf());
                trn_lbl.push(i);
            } else {
                tst_img.push(entry.path().to_path_buf());
                tst_lbl.push(i);
            }
        }
    }

    let trn_size = trn_img.len();
    let tst_size = tst_img.len();


    Ok(TrainingData {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        trn_size,
        tst_size,
        rows,
        cols,
        classes,
    })
}

pub fn load_50states2k<T>(base_path: T, filter: Option<Vec<&str>>) -> Result<TrainingData, String>
where T: AsRef<Path>
{
    let mut img = Vec::new();
    let mut lbl = Vec::new();
    let rows = 256;
    let cols = 256;
    let mut classes: HashMap<usize, usize> = HashMap::new();
    
    for i in 0..50 {
        if let Some(filter) = &filter {
            if !filter.contains(&idx_to_state(i)) {
                continue;
            }
        }
        classes.insert(i, classes.len());
        let state = idx_to_state(i);
        let folder_path = base_path.as_ref().join("50States2K").join(state);
        if !folder_path.exists() {
            return Err(format!("Folder {} does not exist.", folder_path.to_str().unwrap()));
        }
        
        for entry in WalkDir::new(folder_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file() && e.path().extension().map(|s| s == "jpg").unwrap_or(false))
        {
            img.push(entry.path().to_path_buf());
            lbl.push(i);
        }
    }

    let tst_size = img.len();

    Ok(TrainingData {
        trn_img: vec![],
        trn_lbl: vec![],
        tst_img: img,
        tst_lbl: lbl,
        trn_size: 0,
        tst_size,
        rows,
        cols,
        classes,
    })
}

pub fn get_random_image(data: &TrainingData) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = data.trn_img.iter().zip(data.trn_lbl.iter()).choose(&mut rng).unwrap();
    let img = load_image(img).unwrap();
    (img, *label)
}

pub fn get_random_test_image(data: &TrainingData) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = data.tst_img.iter().zip(data.tst_lbl.iter()).choose(&mut rng).unwrap();
    let img = load_image(img).unwrap();
    (img, *label)
}