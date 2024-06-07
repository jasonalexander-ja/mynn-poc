use super::{activations::Activation, matrix::Matrix};

pub trait Layer<const COLS: usize, const END_S: usize> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, act: &Activation<'a>) -> [f64; END_S];

    fn back_propagate<'a>(&mut self, lrate: f64, outputs: [f64; END_S], targets: [f64; END_S], act: &Activation<'a>) -> BackProps<COLS>;
}


pub struct ProcessLayer<const ROWS: usize, const COLS: usize, const END_S: usize, T: Layer<ROWS, END_S>> {
    pub next: T,
    pub weights: Matrix<ROWS, COLS>,
    pub biases: Matrix<ROWS, 1>,
    pub data: Matrix<COLS, 1>
}

impl <const ROWS: usize, const COLS: usize, const END_S: usize, T: Layer<ROWS, END_S>> ProcessLayer<ROWS, COLS, END_S, T> {
    pub fn new(next: T) -> ProcessLayer<ROWS, COLS, END_S, T> {
        ProcessLayer {
            next,
            weights: Matrix::random(),
            biases: Matrix::random(),
            data: Matrix::zeros()
        }
    }

    pub fn predict<'a>(&mut self, data: [f64; COLS], act: &Activation<'a>) -> [f64; END_S] {
        self.feed_forward(Matrix::from([data]).transpose(), act)
    }

    pub fn train<'a, const DATA_S: usize>(&mut self, lrate: f64, inputs: [[f64; COLS]; DATA_S], targets: [[f64; END_S]; DATA_S], epochs: usize, act: &Activation<'a>) {
        for _ in 1..=epochs {
            for i in 0..DATA_S {
                let outputs = self.feed_forward(Matrix::from([inputs[i]]).transpose(), act);
                self.back_propagate(lrate, outputs, targets[i].clone(), act);
            }
        }
    }

}

impl <const ROWS: usize, const COLS: usize, const END_S: usize, T: Layer<ROWS, END_S>> Layer<COLS, END_S> for ProcessLayer<ROWS, COLS, END_S, T> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, act: &Activation<'a>) -> [f64; END_S] {
        self.data = feed;
        let result = self.weights.multiply(&self.data)
            .add(&self.biases)
            .map(act.function);
        self.next.feed_forward(result, act)
    }

    fn back_propagate<'a>(&mut self, lrate: f64, outputs: [f64; END_S], targets: [f64; END_S], act: &Activation<'a>) -> BackProps<COLS> {
        let BackProps(errors, gradients) = self.next.back_propagate(lrate, outputs, targets, act);
        let gradients = gradients.dot_multiply(&errors).map(&|x| x * lrate);

        self.weights = self.weights.add(&gradients.multiply(&self.data.transpose()));
        self.biases = self.biases.add(&gradients);

        let errors = self.weights.transpose().multiply(&errors);
        let gradients = self.data.map(&act.derivative);

        BackProps(errors, gradients)
    }
}


pub struct EndLayer<const END_S: usize>();

impl <const END_S: usize> Layer<END_S, END_S> for EndLayer<END_S> {
    fn feed_forward<'a>(&mut self, feed: Matrix<END_S, 1>, _act: &Activation<'a>) -> [f64; END_S] {
        feed.transpose().data[0]
    }

    fn back_propagate<'a>(&mut self, _lrate: f64, outputs: [f64; END_S], targets: [f64; END_S], act: &Activation<'a>) -> BackProps<END_S> {
        let parsed = Matrix::from([outputs]).transpose();
        let errors = Matrix::from([targets]).transpose().subtract(&parsed);
        let gradients = parsed.map(&act.derivative);
        BackProps(errors, gradients)
    }
}

pub struct BackProps<const COLS: usize>(Matrix<COLS, 1>, Matrix<COLS, 1>);
