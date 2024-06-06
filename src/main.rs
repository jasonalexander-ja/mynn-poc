pub mod matrix;
pub mod network;
pub mod activations;

use network::{Layer, ProcessLayer, EndLayer}; 
use matrix::Matrix; 
use activations::SIGMOID;

fn main() {
    let mut network: ProcessLayer<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = ProcessLayer::new(ProcessLayer::new(EndLayer()));

    let inputs: Vec<[f64; 2]> = vec![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    let targets: Vec<[f64; 1]> = vec![
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ];

    let epochs: u32 = 10_000;

    for i in 1..=epochs {
        if epochs < 100 || i % (epochs / 100) == 0 {
            println!("Epoch {} of {}", i , epochs);
        }

        for j in 0..inputs.len() {
            let outputs = network.feed_forward(Matrix::from([[inputs[j][0]], [inputs[j][1]]]), SIGMOID);
            network.back_propagate(0.5, outputs, targets[j].clone(), &SIGMOID);
        }
    }

    println!("0 and 0: {:?}", network.feed_forward(Matrix::from([[0.0], [0.0]]), SIGMOID));
    println!("1 and 0: {:?}", network.feed_forward(Matrix::from([[1.0], [0.0]]), SIGMOID));
    println!("0 and 1: {:?}", network.feed_forward(Matrix::from([[0.0], [1.0]]), SIGMOID));
    println!("1 and 1: {:?}", network.feed_forward(Matrix::from([[1.0], [1.0]]), SIGMOID));
}

#[cfg(test)]
mod tests {
    use super::*;

    use network::{Layer, ProcessLayer, EndLayer}; 
    use matrix::Matrix; 
    use activations::SIGMOID;

    #[test]
    fn run() {
        let mut network: ProcessLayer<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = ProcessLayer::new(ProcessLayer::new(EndLayer()));

        let inputs: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
    
        let targets: Vec<[f64; 1]> = vec![
            [0.0],
            [1.0],
            [1.0],
            [1.0],
        ];
    
        let epochs: u32 = 10_000;
    
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i , epochs);
            }
    
            for j in 0..inputs.len() {
                let outputs = network.feed_forward(Matrix::from([[inputs[j][0]], [inputs[j][1]]]), SIGMOID);
                network.back_propagate(0.5, outputs, targets[j].clone(), &SIGMOID);
            }
        }
    
        println!("0 and 0: {:?}", network.feed_forward(Matrix::from([[0.0], [0.0]]), SIGMOID));
        println!("1 and 0: {:?}", network.feed_forward(Matrix::from([[1.0], [0.0]]), SIGMOID));
        println!("0 and 1: {:?}", network.feed_forward(Matrix::from([[0.0], [1.0]]), SIGMOID));
        println!("1 and 1: {:?}", network.feed_forward(Matrix::from([[1.0], [1.0]]), SIGMOID));
    }
}
