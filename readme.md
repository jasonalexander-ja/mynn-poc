# Heap-less neural network POC 

This is a simple proof of concept of how a type-safe ANN could be made using all stack allocated memory and therefore in theory be used without the standard library and thus usable by not only application crates but also embedded crates for TinyML. 

__Disclaimer: I am in no way a qualified or experienced data scientist or a specialist in the subject of ML, I simply saw an interesting approach to representing simple neural networks, suggestions and PRs from anyone who knows what they're doing is very welcome.__ 

## Explanation 

Here I go over the structure of the neural network I have used in this POC from the very basics (I.E. assuming no knowledge of neural networks from the reader) to better explain the data structures I used later. 

### Neural Networks 

Neural networks are described as consisting of "layers" of interconnected neurons. 

<img src="docs/231NN.png">

This is represented in POC by each layer having two 2-dimensional arrays of floating point numbers called "matrices" (singular, matrix), matrices have a number of columns (inner array length), and rows (outer array length). 

The two matrices in each layer is called the weights and biases, they will both have equal numbers of columns that corresponds to the number of neurons in that layer; however the number of rows in the biases will always be equal to 1, but the number of rows in the weights must be equal to the number of neurons (columns) in the next layer, this is due to the fact that when matrices are multiplied together (remember that all the neurons are interconnected, this is what that is representing) the number of rows in the first matrix must equal the number of columns in the second. 

Thus, the neural network in the diagram above will look something like this:

```
------- First Layer -------
Weights: 
[
    [0, 0],
    [0, 0],
    [0, 0],
]

Biases:
[
    [0, 0]
]
------- Second Layer ------
Weights:
[
    [0, 0, 0]
]
Biases:
[
    [0, 0, 0]
]
```

Since the final layer is just the output, that will just simply be the result of the 2nd layer and doesn't actually "exist". 

### The Data Structures 


#### Matrices 

In most programming languages the most common way to represent a matrix would be to use a 2 dimensional dynamically sized array or vector, using runtime parameters to define and enforce it's size and shape, for instance a common representation in Rust may be;

```rust
struct Matrix(Vec<Vec<f64>>); 

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix(vec![vec![0.0; cols]; rows])
    } 
}
```

This is good but since vectors require some kind of heap memory assignment, we can't really use this with `no_std` which is required if we want our code to work absolute anywhere including on embedded systems. 

To do this, we must only use data structures that have a known size at compile time, or fixed sized arrays, in other languages this would present an issue since for each possible size of a matrix we'd need to make an entirely new matrix type, and then write custom code over and over again defining how that matrix can multiply, add, and subtract from other compatible matrix types - quite verbose. 

Thankfully, Rust has a concept called const generics; think how vectors can accept a wide range of different types without having to define a new vector type for each possible type because they accept a generic type parameter, in Rust, we can specify compile-time known constant values in our type parameters such as unsigned integers to our custom matrix type, specifying the number of rows and columns, that we can then pass on to a fixed sized array: 

```rust
struct Matrix<const ROWS: usize, const COLS: usize>([[f64; COLS]; ROWS]);

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn new() -> Matrix {
        Matrix([[0.0; COLS]; ROWS])
    } 
}
```
We can then also specify type bounds on how matrices interact with one and other, for instance remember that when multiplying 2 matrices, the number of columns in one matrix must equal the number of rows in the other, we can ensure this at compile time via:

```rust

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {

    pub fn multiply<const OTHER_COLS: usize>(&self, other: Matrix<COLS, OTHER_COLS>) -> Matrix<ROWS, OTHER_COLS> {
        ...
    }

}

```

Because the size of a matrix is now in it's type, we can make sure that operations between correctly sized matrices happen at compile time via the type system, we can see above in the method definition that the other matrix must have a number of rows equal to the number of columns to the matrix it's being called on. 

Now that we have fixed sized, stack allocated matrices, we can now look at how to build a neural network from them. 

#### Stack-based Neural Network Types

Since neural networks are made up of layers, this feels like the best place to start; our layer type will just be a simple struct that has 2 matrices, the weights and biases, similar to the matrix, we will also have generic const parameters for the neurons in that layer (the number of columns for both the matrices) and the neurons in the next layer (the number of rows in the weights matrix). 

```rust

struct Layer<const ROWS: usize, const NEURONS: usize> {
    weights: Matrix<ROWS, NEURONS>,
    biases: Matrix<1, NEURONS> // biases always has 1 row 
}

```

The neural network I used as a reference for this stored all of it's layers in a collection type, this would mean all of our layers would have to be the exact same size since they would need to be the exact same type, this isn't ideal, a better way of doing this would be to make our layers a linked list, since linked lists require at least 2 types so we know where the trail ends, we can make our layer into a trait with 2 inheritors: 

```rust

trait Layer {}


struct ActiveLayer<const ROWS: usize, const NEURONS: usize, T: Layer> {
    next: T,
    weights: Matrix<ROWS, NEURONS>,
    biases: Matrix<1, NEURONS> // biases always has 1 row 
}

impl<const ROWS: usize, const NEURONS: usize, T: Layer> Layer for ActiveLayer<ROWS, NEURONS, T> {}


struct EndLayer();

impl Layer for EndLayer {}

```

This is great! We also want to make sure that our next layer has the same number of neurons as there are rows in the weights in the current layer, there is a small change we can make to ensure this type safety, we can add a neurons const generic parameter to the layer trait, that way we can pass our rows parameter in the active layer struct into it, enforcing that the next layer must have that number of neurons;

```rust
trait Layer<const NEURONS: usize> {}


struct ActiveLayer<const ROWS: usize, const NEURONS: usize, T: Layer<ROWS>> { 
    ... 
}

impl <const ROWS: usize, const NEURONS: usize, T: Layer<ROWS>> Layer<NEURONS> for ActiveLayer<ROWS, NEURONS, T> {}

```

There we go! We are now type bound to make to structurally correct neural network, just one small adjustment to the end layer; 

```rust
struct EndLayer<const NEURONS: usize>();

impl <const NEURONS: usize> Layer<NEURONS> for EndLayer<NEURONS> {}
```

You can see a [full example of these types in action here](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=1d4495b445e2c432ff20de892286c764), on line 18 you can see a network being built up using these types, you can try for instance changing the number of rows in the first (outer) layer, it will cause a compiler error ;) 

```rust
// Ok
let network: ActiveLayer<3, 2, ActiveLayer<1, 3, EndLayer<1>>> = ActiveLayer(ActiveLayer(EndLayer()));

// Compiler error!
let network: ActiveLayer<1, 2, ActiveLayer<1, 3, EndLayer<1>>> = ActiveLayer(ActiveLayer(EndLayer()));
```

And we're done, the only difference between the code in this explanation and the full POC in this repository is some small additions for easier operation for feeding forward and back propagation, but this shows all of the important features that allow for entirely stack-memory based operation and type safety! 

### The POC 

There are a few differences between the POC and the examples types above. 

Firstly and most obvious is that the layer trait has and a const generic parameter named `END_S`, this is the number of neurons on the last layer, this is thus present on all layers and means that when we are forward feeding (I.E. predicting) and back propagating (I.E. correcting) we know the size of the array being passed back and forward is;

```rust
use super::{activations::Activation, matrix::Matrix};

pub trait Layer<const COLS: usize, const END_S: usize> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, act: &Activation<'a>) -> [f64; END_S];

    fn back_propagate<'a>(&mut self, lrate: f64, outputs: [f64; END_S], targets: [f64; END_S], act: &Activation<'a>) -> BackProps<COLS>;
}


pub struct ProcessLayer<const ROWS: usize, const COLS: usize, const END_S: usize, T: Layer<ROWS, END_S>> {
    ...
}

impl <const ROWS: usize, const COLS: usize, const END_S: usize, T: Layer<ROWS, END_S>> Layer<COLS, END_S> for ProcessLayer<ROWS, COLS, END_S, T> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, act: &Activation<'a>) -> [f64; END_S] {
        ...
    }
    fn back_propagate<'a>(&mut self, lrate: f64, outputs: [f64; END_S], targets: [f64; END_S], act: &Activation<'a>) -> BackProps<COLS> {
        ...
    }
}


pub struct EndLayer<const END_S: usize>();

impl <const END_S: usize> Layer<END_S, END_S> for EndLayer<END_S> {
    fn feed_forward<'a>(&mut self, feed: Matrix<END_S, 1>, _act: &Activation<'a>) -> [f64; END_S] {
        feed.transpose().data[0]
    }
    fn back_propagate<'a>(&mut self, _lrate: f64, outputs: [f64; END_S], targets: [f64; END_S], act: &Activation<'a>) -> BackProps<END_S> {
        ...
    }
}

pub struct BackProps<const COLS: usize>(Matrix<COLS, 1>, Matrix<COLS, 1>);

```

The methods `feed_forward` and `back_propagate` are how we use the model to make predictions and run corrections, the details of how they work aren't too relevant to this POC. 

The final difference is in the process layer struct, there is an extra field in there named `data` that contains the data that layer was fed from the latest feed forward, this is important to store when running `back_propagate` and we are making corrections. 

```rust
pub struct ProcessLayer<const ROWS: usize, const COLS: usize, const END_S: usize, T: Layer<ROWS, END_S>> {
    pub next: T,
    pub weights: Matrix<ROWS, COLS>,
    pub biases: Matrix<ROWS, 1>,
    pub data: Matrix<COLS, 1>
}
```

#### Using the POC 

To instantiate the proof of concept network, this is similar to the example from the types example above just that we need to specify the final layer size in all of our layers until the end layer, for instance a 2, 3, 1 neural network like the example above is: 

```rust 
let mut network: ProcessLayer<3, 2, 1, ProcessLayer<1, 3, 1, EndLayer<1>>> = 
    ProcessLayer::new(ProcessLayer::new(EndLayer()));
```

We can then set up some simple training data, for instance we can try to predict the output of a 2 input XOR gate, this has 4 possible states so we just need 4 data sets;

```rust
let inputs: [[f64; 2]; 4] = [
    [0.0, 0.0], [0.0, 1.0],
    [1.0, 0.0], [1.0, 1.0],
];
let targets: [[f64; 1]; 4] = [ [0.0], [1.0], [1.0], [0.0], ];
```

We can then feed this into our network above using the train method, specifying a learning rate, the number of epochs (the number of times it will try to predict the data and go back and make corrections), and an activation function (a sigmoid activation function is provided). In this case we specify a learning rate of 0.5 and 10,000 epochs. 

```rust
network.train(0.5, inputs, targets, 10_000, &SIGMOID);

```

Awesome, now we can see how our neural network did using the predict method and specifying the same activation function and giving it some data: 

```rust
println!("0 and 0: {:?}", network.predict([0.0, 0.0], &SIGMOID));
println!("1 and 0: {:?}", network.predict([1.0, 0.0], &SIGMOID));
println!("0 and 1: {:?}", network.predict([0.0, 1.0], &SIGMOID));
println!("1 and 1: {:?}", network.predict([1.0, 1.0], &SIGMOID));
```

We should see the outputs along the lines of:

```
0 and 0: [0.009464095466212581]
1 and 0: [0.9878240431204596]
0 and 1: [0.9878248964508319]
1 and 1: [0.014166198831006123]
```

Great! Neural networks are never 100% certain so this is expected. 

