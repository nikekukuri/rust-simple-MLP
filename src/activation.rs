#[derive(Debug)]
pub enum Activation {
    Sigmoid,
    Relu,
}

impl Activation {
    pub fn forward(&self) -> fn(f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid,
            Self::Relu => relu,
        }
    }

    pub fn backward(&self) -> fn(f64) -> f64 {
        match self {
            Self::Sigmoid => sigmoid_deriv,
            Self::Relu => relu_deriv,
        }
    }
}

fn relu(x: f64) -> f64 {
    if x < 0. {
        0.
    } else {
        x
    }
}

fn relu_deriv(y: f64) -> f64 {
    if y < 0. { 
        0.
    } else {
        1.
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_deriv(y: f64) -> f64 {
    y * (1.0 - y)
}
