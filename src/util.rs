use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use image::io::Reader as ImageReader;
use ndarray::Array3;

// Offer on-demand loading of images and full dataset loading
#[derive(Serialize, Deserialize, Clone)]
pub enum TrainImage {
    Image(Array3<f32>),
    Path(PathBuf),
}

#[derive(Serialize, Deserialize, Default)]
pub struct TrainingData {
    pub trn_img: Vec<TrainImage>,
    pub trn_lbl: Vec<usize>,
    pub tst_img: Vec<TrainImage>,
    pub tst_lbl: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub trn_size: usize,
    pub tst_size: usize,
    pub classes: HashMap<usize, usize>,
}

/// Computes the outer product of two vectors
pub fn outer(x: Array1<f32>, y: Array1<f32>) -> Array2<f32> {
    let mut result: Array2<f32> = Array2::<f32>::zeros((x.len(), y.len()));

    for i in 0..x.len() {
        for j in 0..y.len() {
            result[[i, j]] = x[i] * y[j];
        }
    }

    result
}

/// Converts a state name to an index, based on
/// alphabetical order of the states
pub fn state_to_idx<T>(state: T) -> usize
where T: Into<&'static str> {
    let state_map: HashMap<&'static str, usize> = [
        ("Alabama", 0), ("Alaska", 1), ("Arizona", 2), ("Arkansas", 3), ("California", 4),
        ("Colorado", 5), ("Connecticut", 6), ("Delaware", 7), ("Florida", 8), ("Georgia", 9),
        ("Hawaii", 10), ("Idaho", 11), ("Illinois", 12), ("Indiana", 13), ("Iowa", 14),
        ("Kansas", 15), ("Kentucky", 16), ("Louisiana", 17), ("Maine", 18), ("Maryland", 19),
        ("Massachusetts", 20), ("Michigan", 21), ("Minnesota", 22), ("Mississippi", 23), ("Missouri", 24),
        ("Montana", 25), ("Nebraska", 26), ("Nevada", 27), ("New Hampshire", 28), ("New Jersey", 29),
        ("New Mexico", 30), ("New York", 31), ("North Carolina", 32), ("North Dakota", 33), ("Ohio", 34),
        ("Oklahoma", 35), ("Oregon", 36), ("Pennsylvania", 37), ("Rhode Island", 38), ("South Carolina", 39),
        ("South Dakota", 40), ("Tennessee", 41), ("Texas", 42), ("Utah", 43), ("Vermont", 44),
        ("Virginia", 45), ("Washington", 46), ("West Virginia", 47), ("Wisconsin", 48), ("Wyoming", 49),
    ].iter().cloned().collect();

    *state_map.get(state.into()).unwrap()
}


/// Converts a state index to a name, based on
/// alphabetical order of the states
pub fn idx_to_state(idx: usize) -> &'static str {
    let cluster_map: HashMap<usize, &'static str> = [
        (0, "Alabama"), (1, "Alaska"), (2, "Arizona"), (3, "Arkansas"), (4, "California"),
        (5, "Colorado"), (6, "Connecticut"), (7, "Delaware"), (8, "Florida"), (9, "Georgia"),
        (10, "Hawaii"), (11, "Idaho"), (12, "Illinois"), (13, "Indiana"), (14, "Iowa"),
        (15, "Kansas"), (16, "Kentucky"), (17, "Louisiana"), (18, "Maine"), (19, "Maryland"),
        (20, "Massachusetts"), (21, "Michigan"), (22, "Minnesota"), (23, "Mississippi"), (24, "Missouri"),
        (25, "Montana"), (26, "Nebraska"), (27, "Nevada"), (28, "New Hampshire"), (29, "New Jersey"),
        (30, "New Mexico"), (31, "New York"), (32, "North Carolina"), (33, "North Dakota"), (34, "Ohio"),
        (35, "Oklahoma"), (36, "Oregon"), (37, "Pennsylvania"), (38, "Rhode Island"), (39, "South Carolina"),
        (40, "South Dakota"), (41, "Tennessee"), (42, "Texas"), (43, "Utah"), (44, "Vermont"),
        (45, "Virginia"), (46, "Washington"), (47, "West Virginia"), (48, "Wisconsin"), (49, "Wyoming"),
    ].iter().cloned().collect();

    cluster_map.get(&idx).unwrap()
}

#[derive(Serialize, Deserialize)]
/// Defines when the model should be saved.
/// The bool is whether to save the full model (true), or just metadata (false)
pub enum SavingStrategy {
    EveryEpoch(bool),
    EveryNthEpoch(bool, f32),
    BestTrainingAccuracy(bool),
    BestTestingAccuracy(bool),
    Never,
}

pub fn load_image(path: &Path) -> Result<Array3<f32>, String> {
    let img = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;
    let img = img.to_rgb8();

    let rows = img.height() as usize;
    let cols = img.width() as usize;
    let mut array = Array3::zeros((rows, cols, 3));

    for (x, y, pixel) in img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        array[[x as usize, y as usize, 0]] = r / 255.0;
        array[[x as usize, y as usize, 1]] = g / 255.0;
        array[[x as usize, y as usize, 2]] = b / 255.0;
    }

    Ok(array)
}