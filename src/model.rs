#![allow(non_snake_case)]
use ndarray::*;
use crate::layer::Layer;

pub struct Model {
    layers: Vec<Layer>,
    pub loss_history: Vec<f64>,
}

impl Model {
    pub fn new(
        layers: Vec<Layer>,
    ) -> Self {
        Self {
            layers,
            loss_history: Vec::new(),
        }
    }

    fn validate_nodes(&self) {
        for i in 1..self.layers.len() {
            if self.layers[i - 1].out.len() != self.layers[i].x.len() {
                panic!("Invalid Nodes Connect. Check layer parameter.");
            }
        }
    }

    pub fn learn_iter(&mut self, X: Vec<Array1<f64>>, t: Vec<Array1<f64>>, epochs: usize) {
        for n in 0..epochs {
            for (x, t) in X.iter().zip(t.iter()) {
                self.learn(x.clone(), t.clone());
            }
            println!("epoch: {}, loss: {:?}", n, self.loss_history.last().unwrap());
        }
    }
    pub fn learn(&mut self, x: Array1<f64>, t: Array1<f64>) {
        self.validate_nodes();
        let loss = self.loss(x, t);
        self.backward(loss);
    }

    pub fn loss(&mut self, x: Array1<f64>, y_train: Array1<f64>) -> Array1<f64> {
        let y_predict = self.forward(x);
        
        // Root Mean Squared Error
        let err_value: Vec<_> = y_predict.iter().zip(y_train.iter()).map(|(a, b)| (a - b).powf(2.)).collect();


        // Cross Entropy Error
        //let err_value = y_predict.iter().map(|y| -y.log10()).collect();
        //let err_value: Vec<_> = y_predict.iter().zip(y_train.iter()).map(|(y, t)| -t * y.log(2.) - (1. - t) * (1. - y).log(2.)).collect();
        
        self.loss_history.push(err_value.iter().sum::<f64>().sqrt());

        Array1::from_vec(err_value)
    }

    pub fn predict(&mut self, x: Array1<f64>) -> Array1<f64> {
        self.forward(x)
    }

    pub fn forward(&mut self, x: Array1<f64>) -> Array1<f64> {
        let mut out = x;
        for layer in self.layers.iter_mut() {
            out = layer.forward(out);
        }
        out
    }

    pub fn backward(&mut self, mut dy: Array1<f64>) {
        for layer in self.layers.iter_mut().rev() {
            dy = layer.backward(dy);
        }
    }

}
