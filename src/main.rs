pub mod matrix;
pub mod network;
pub mod activations;

use network::{Layer, ProcessLayer, EndLayer}; 
use matrix::Matrix; 
use activations::SIGMOID;

fn main() {

    let inputs: [[f64; 2]; 4] = [
        [0.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [1.0, 1.0],
    ];
    let targets: [[f64; 1]; 4] = [ [0.0], [1.0], [1.0], [0.0], ];

    let mut network: ProcessLayer<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = ProcessLayer::new(ProcessLayer::new(EndLayer()));

    for _ in 1..=10_000 {
        for i in 0..4 {
            let outputs = network.feed_forward(Matrix::from([[inputs[i][0]], [inputs[i][1]]]), &SIGMOID);
            network.back_propagate(0.5, outputs, targets[i].clone(), &SIGMOID);
        }
    }
}

