use ndarray::{Array1, Array2, Array4};
use serde::{Serialize, Deserialize};
use std::fmt::{Debug, Formatter};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum OptimizerAlg {
    SGD(f32),
    Momentum(f32, f32),
    RMSProp(f32, f32),
    Adam(f32, f32, f32),
}

impl Debug for OptimizerAlg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        match self {
            OptimizerAlg::SGD(lr) => {
                s.push_str(&format!("SGD\n"));
                s.push_str(&format!(" - Learning Rate: {}\n", lr));
            },
            OptimizerAlg::Momentum(lr, mu) => {
                s.push_str(&format!("Momentum\n"));
                s.push_str(&format!(" - Learning Rate: {}\n", lr));
                s.push_str(&format!(" - Momentum: {}\n", mu));
            },
            OptimizerAlg::RMSProp(lr, rho) => {
                s.push_str(&format!("RMSProp\n"));
                s.push_str(&format!(" - Learning Rate: {}\n", lr));
                s.push_str(&format!(" - Rho: {}\n", rho));
            },
            OptimizerAlg::Adam(lr, beta1, beta2) => {
                s.push_str(&format!("Adam\n"));
                s.push_str(&format!(" - Learning Rate: {}\n", lr));
                s.push_str(&format!(" - Beta1: {}\n", beta1));
                s.push_str(&format!(" - Beta2: {}\n", beta2));
                s.push_str(&format!(" - Epsilon: {}\n", 1e-8));
            },
        }

        write!(f, "{}", s)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Optimizer2D {
    pub alg: OptimizerAlg,
    pub momentum1: Array2<f32>,
    pub momentum2: Array2<f32>,
    pub t: i32,
    pub beta1_done: bool,
    pub beta2_done: bool,
}

impl Optimizer2D {
    pub fn new(alg: OptimizerAlg, input_size: usize, output_size: usize) -> Optimizer2D {
        let momentum1 = Array2::<f32>::zeros((output_size, input_size));
        let momentum2 = Array2::<f32>::zeros((output_size, input_size));
        let t = 0;
        let beta1_done = false;
        let beta2_done = false;

        Optimizer2D {
            alg,
            momentum1,
            momentum2,
            t,
            beta1_done,
            beta2_done,
        }
    }

    pub fn weight_changes(&mut self, gradients: &Array2<f32>) -> Array2<f32> {
        match self.alg {
            OptimizerAlg::SGD(lr) => {
                gradients * lr
            },
            OptimizerAlg::Momentum(lr, mu) => {
                self.momentum1 = &self.momentum1 * mu + gradients;
                &self.momentum1 * lr
            },
            OptimizerAlg::RMSProp(lr, rho) => {
                self.momentum1 = &self.momentum1 * rho;
                self.momentum1 += &(gradients.mapv(|x| x.powi(2) as f32) * (1.0 - rho));
                gradients * lr / (self.momentum1.mapv(|x| x.sqrt()) + 1e-8)
            },
            OptimizerAlg::Adam(lr, beta1, beta2) => {
                self.t += 1;
                self.momentum1 = &self.momentum1 * beta1;
                self.momentum1 += &(gradients.mapv(|x| x * (1.0 - beta1)));
                self.momentum2 = &self.momentum2 * beta2;
                self.momentum2 += &(gradients.mapv(|x| x.powi(2) * (1.0 - beta2) as f32));
                let biased_beta1 = if self.beta1_done {
                    0.0
                } else {
                    let pow = beta1.powi(self.t);
                    if pow < 0.001 {
                        self.beta1_done = true;
                    }
                    pow
                };
                let biased_beta2 = if self.beta2_done {
                    0.0
                } else {
                    let pow = beta2.powi(self.t);
                    if pow < 0.001 {
                        self.beta2_done = true;
                    }
                    pow
                };

                let weight_velocity_corrected = &self.momentum1 / (1.0 - biased_beta1);
                let weight_velocity2_corrected = &self.momentum2 / (1.0 - biased_beta2);
                &weight_velocity_corrected * lr / (weight_velocity2_corrected.mapv(|x| x.sqrt()) + 1e-8)
            },
        }
    }

    pub fn bias_changes(&mut self, gradients: &Array1<f32>) -> Array1<f32> {
        match self.alg {
            OptimizerAlg::SGD(lr) => {
                gradients * lr
            },
            OptimizerAlg::Momentum(lr, _) => {
                gradients * lr
            },
            OptimizerAlg::RMSProp(lr, _) => {
                gradients * lr
            },
            OptimizerAlg::Adam(lr, _, _) => {
                gradients * lr
            },
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Optimizer4D {
    pub alg: OptimizerAlg,
    pub momentum1: Array4<f32>,
    pub momentum2: Array4<f32>,
    pub t: i32,
    pub beta1_done: bool,
    pub beta2_done: bool,
}

impl Optimizer4D {
    pub fn new(alg: OptimizerAlg, size: (usize, usize, usize, usize)) -> Optimizer4D {
        let momentum1 = Array4::<f32>::zeros(size);
        let momentum2 = Array4::<f32>::zeros(size);
        let t = 0;
        let beta1_done = false;
        let beta2_done = false;

        Optimizer4D {
            alg,
            momentum1,
            momentum2,
            t,
            beta1_done,
            beta2_done,
        }
    }

    pub fn weight_changes(&mut self, gradients: &Array4<f32>) -> Array4<f32> {
        match self.alg {
            OptimizerAlg::SGD(lr) => {
                gradients * lr
            },
            OptimizerAlg::Momentum(lr, mu) => {
                self.momentum1 = &self.momentum1 * mu + gradients;
                &self.momentum1 * lr
            },
            OptimizerAlg::RMSProp(lr, rho) => {
                self.momentum1 = &self.momentum1 * rho;
                self.momentum1 += &(gradients.mapv(|x| x.powi(2) as f32) * (1.0 - rho));
                gradients * lr / (self.momentum1.mapv(|x| x.sqrt()) + 1e-8)
            },
            OptimizerAlg::Adam(lr, beta1, beta2) => {
                self.t += 1;
                self.momentum1 = &self.momentum1 * beta1;
                self.momentum1 += &(gradients.mapv(|x| x * (1.0 - beta1)));
                self.momentum2 = &self.momentum2 * beta2;
                self.momentum2 += &(gradients.mapv(|x| x.powi(2) * (1.0 - beta2) as f32));
                let biased_beta1 = if self.beta1_done {
                    0.0
                } else {
                    let pow = beta1.powi(self.t);
                    if pow < 0.001 {
                        self.beta1_done = true;
                    }
                    pow
                };
                let biased_beta2 = if self.beta2_done {
                    0.0
                } else {
                    let pow = beta2.powi(self.t);
                    if pow < 0.001 {
                        self.beta2_done = true;
                    }
                    pow
                };

                let weight_velocity_corrected = &self.momentum1 / (1.0 - biased_beta1);
                let weight_velocity2_corrected = &self.momentum2 / (1.0 - biased_beta2);
                &weight_velocity_corrected * lr / (weight_velocity2_corrected.mapv(|x| x.sqrt()) + 1e-8)
            },
        }
    }
}