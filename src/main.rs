pub mod matrix;
pub mod network;
pub mod activations;

use network::{EndLayer, ProcessLayer};
use activations::SIGMOID;

fn main() {

    let inputs: [[f64; 2]; 4] = [
        [0.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [1.0, 1.0],
    ];
    let targets: [[f64; 1]; 4] = [ [0.0], [1.0], [1.0], [0.0], ];

    let mut network: ProcessLayer<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = 
        ProcessLayer::new(ProcessLayer::new(EndLayer()));

    network.train(0.5, inputs, targets, 10_000, &SIGMOID);

    println!("0 and 0: {:?}", network.predict([0.0, 0.0], &SIGMOID));
    println!("1 and 0: {:?}", network.predict([1.0, 0.0], &SIGMOID));
    println!("0 and 1: {:?}", network.predict([0.0, 1.0], &SIGMOID));
    println!("1 and 1: {:?}", network.predict([1.0, 1.0], &SIGMOID));
}
