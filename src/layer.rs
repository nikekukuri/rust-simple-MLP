use ndarray::*;
use ndarray_linalg::*;
use rand::Rng;

use crate::activation::Activation;

#[derive(Debug)]
pub struct Layer {
    activation: Activation,
    bias: Array1<f64>,
    dbias: Array1<f64>,
    weight: Array2<f64>,
    dweight: Array2<f64>,
    pub x: Array1<f64>,
    pub out: Array1<f64>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let shape = (input_size, output_size);

        let mut weight_vec = Vec::new();

        // Initialize weights at random
        for _ in 0..shape.0 * shape.1 {
            weight_vec.push(rand::thread_rng().gen_range(0.0..1.0));
        }

        Self {
            activation,
            bias: Array1::zeros(output_size),
            dbias: Array1::zeros(output_size),

            weight: Array::from_shape_vec(shape, weight_vec).unwrap(),
            dweight: Array::zeros(shape),
            x: Array1::zeros(input_size),
            out: Array1::zeros(output_size),
        }
    }

    pub fn forward(&mut self, x: Array1<f64>) -> Array1<f64> {
        self.x.clone_from(&x);
        let z = x.dot(&self.weight) + &self.bias;
        let act_fn = self.activation.forward();

        let out = z.mapv(act_fn);
        self.out.clone_from(&out);
        out
    }

    pub fn backward(&mut self, dy: Array1<f64>) -> Array1<f64> {
        let learning_rate = 0.001;  //TODO: read external file
        let act_fn = self.activation.backward();
        let dz = dy * self.out.mapv(act_fn);
        let dx = dz.clone().dot(&self.weight.t());

        // Calcurate gradient
        self.dweight = self.x.clone().insert_axis(Axis(1)).dot(&dz.clone().insert_axis(Axis(0)));
        self.dbias = dz;

        // Update weight & bias
        self.weight = &self.weight - (learning_rate * &self.dweight);
        self.bias = &self.bias - (learning_rate * &self.dbias);

        dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_shape() {
        let layer = Layer::new(3, 2, Activation::Relu);
        assert_eq!(layer.weight.shape(), &[3, 2]);
    }

    #[test]
    fn test_layer_forward() {
        let mut layer = Layer::new(3, 2, Activation::Relu);
        layer.weight = array![[1., 1.], [1., 1.], [1., 1.]];
        let a = array![2., 2., 2.];
        let result = layer.forward(a);
        assert_eq!(result, array![6., 6.]);
        assert_eq!(&layer.x, array![2., 2., 2.]);
    }

    #[test]
    fn test_layer_backward_shape() {
        let mut layer = Layer::new(3, 2, Activation::Relu);
        layer.weight = array![[1., 1.], [1., 1.], [1., 1.]];
        let a = array![2., 2., 2.];
        let dy = array![2., 2.];
        let _ = layer.forward(a);
        let result = layer.backward(dy);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_layer_backward() {
        let mut layer = Layer::new(3, 2, Activation::Relu);
        layer.weight = array![[1., 2.], [3., 1.], [1., 2.]];
        let a = array![2., 2., 2.];
        let dy = array![2., 2.];
        let _ = layer.forward(a);
        let result = layer.backward(dy);
        assert_eq!(result, array![6., 8., 6.]);
    }

    #[test]
    fn test_layer_backward_update_weight() {
        let mut layer = Layer::new(3, 2, Activation::Relu);
        layer.weight = array![[1., 2.], [3., 1.], [1., 2.]];
        let weight_tmp = array![[1., 2.], [3., 1.], [1., 2.]]; 
        let a = array![2., 3., 2.];
        let dy = array![2., 2.];
        let _ = layer.forward(a);
        let _ = layer.backward(dy);
        let dw = layer.dweight;

        let updated_weight = weight_tmp - 0.001 * array![[4., 4.], [6., 6.], [4., 4.]];
        assert_eq!(dw, array![[4., 4.], [6., 6.], [4., 4.]]);
        assert_eq!(layer.weight, updated_weight);
    }
}

