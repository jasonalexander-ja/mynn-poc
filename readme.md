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
