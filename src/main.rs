pub mod matrix;
pub mod network;
pub mod activations;

use network::{EndLayer, ProcessLayer};
use activations::SIGMOID;

macro_rules! network_instantiate {
    ($a:expr, $b:expr) => {
        EndLayer()
    };
    ($a:expr,$($b:tt)*) => {
        ProcessLayer::new(network_instantiate!($($b)*))
    }
}


macro_rules! network_type {
    ($neurons:expr) => {
        EndLayer::<$neurons>
    };
    (end $end_s:expr, $neurons:expr) => {
        ProcessLayer::<$end_s, $neurons, $end_s, EndLayer::<$end_s>>
    };
    (end $end_s:expr, $neurons:expr, $c:expr) => {
        ProcessLayer::<$c, $neurons, $end_s, network_type!(end $end_s, $c)>
    };
    (end $end_s:expr, $neurons:expr, $next:expr, $($d:tt)*) => {
        ProcessLayer::<$next, $neurons, $end_s, network_type!(end $end_s, $next, $($d)*)>
    }
}


macro_rules! network {
    (end $end_s:expr) => {
        network_type!($end_s)()
    };
    (end $end_s:expr, $neurons:expr) => {
        ProcessLayer::<$end_s, $neurons, $end_s, network_type!($end_s)>::new(network_instantiate!($neurons, $end_s))
    };
    (end $end_s:expr, $neurons:expr, $next:expr) => {
        ProcessLayer::<$next, $neurons, $end_s, network_type!(end $end_s, $next)>::new(network_instantiate!($neurons, $next, $end_s))
    };
    (end $end_s:expr, $neurons:expr, $next:expr, $($d:tt)*) => {
        ProcessLayer::<$next, $neurons, $end_s, network_type!(end $end_s, $next, $($d)*)>::new(network_instantiate!($neurons, $next, $($d)*, $end_s))
    }
}

fn main() {


    let inputs: [[f64; 2]; 4] = [
        [0.0, 0.0], 
        [0.0, 1.0],
        [1.0, 0.0], 
        [1.0, 1.0],
    ];
    let targets: [[f64; 1]; 4] = [ 
        [0.0], 
        [1.0], 
        [1.0], 
        [0.0], 
    ];


    let mut network = network!(end 1, 2, 3);

    network.train(0.5, inputs, targets, 10_000, &SIGMOID);


    println!("0 and 0: {:?}", network.predict([0.0, 0.0], &SIGMOID));
    println!("1 and 0: {:?}", network.predict([1.0, 0.0], &SIGMOID));
    println!("0 and 1: {:?}", network.predict([0.0, 1.0], &SIGMOID));
    println!("1 and 1: {:?}", network.predict([1.0, 1.0], &SIGMOID));
}
