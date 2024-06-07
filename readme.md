# Heap-less neural network POC 

This is a simple proof of concept of how an ANN could work without using a heap and therefore in theory be used without the standard library and thus usable by not only application crates but also embedded crates for TinyML. 

__Disclaimer: I am in no way a qualified or experienced data scientist or a specialist in the subject of ML, I simply saw an interesting approach to representing simple neural networks.__ 

## Explaination 

Here I go over the basics of the structure of a neural network in code from the very basics to better explain the data structures I used later. 

### Neural Networks 

Neural networks are described as consisting of "layers" of interconnected neurons. 

<img src="docs/231NN.png">

This is commonly represented in code by each layer having two 2-dimensional arrays of floating point numbers called "matrices" (singular, matrix), matrices have a number of columns (inner array length), and rows (outer array length). 

The two matrices in each layer is called the weights and biases, they will both have equal numbers of columns that corresponds to the number of neurons in that layer; however the number of rows in the biases will always be equal to 1, but the number of rows in the weights must be equal to the number of neurons (columns) in the next layer, this is due to the fact that when matrices are multiplied together (remember that all the neurons are interconnected, this is what that is representing) the number of rows in the first matrix must equal the number of rows in the second. 

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

This is good but since vectors require some kind of memory assignment system, we can't really use this with `no_std` which removes any kind of memory assignment. 

