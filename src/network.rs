use super::{activations::Activation, matrix::Matrix};

pub trait Layer<const COLS: usize, const TARGET: usize> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, act: Activation<'a>) -> [f64; TARGET];

    fn back_propagate<'a>(&mut self, lrate: f64, outputs: [f64; TARGET], targets: [f64; TARGET], act: &Activation<'a>) -> BackProps<COLS>;
}


pub struct ProcessLayer<const ROWS: usize, const COLS: usize, const TARGET: usize, T: Layer<ROWS, TARGET>> {
    pub next: T,
    pub weights: Matrix<ROWS, COLS>,
    pub biases: Matrix<ROWS, 1>,
    pub data: Matrix<COLS, 1>
}

impl <const ROWS: usize, const COLS: usize, const TARGET: usize, T: Layer<ROWS, TARGET>> ProcessLayer<ROWS, COLS, TARGET, T> {
    pub fn new(next: T) -> ProcessLayer<ROWS, COLS, TARGET, T> {
        ProcessLayer {
            next,
            weights: Matrix::random(),
            biases: Matrix::random(),
            data: Matrix::zeros()
        }
    }
}

impl <const ROWS: usize, const COLS: usize, const TARGET: usize, T: Layer<ROWS, TARGET>> Layer<COLS, TARGET> for ProcessLayer<ROWS, COLS, TARGET, T> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, act: Activation<'a>) -> [f64; TARGET] {
        self.data = feed.clone();
        let result = self.weights.multiply(&feed)
            .add(&self.biases)
            .map(&act.function);
        self.next.feed_forward(result, act)
    }

    fn back_propagate<'a>(&mut self, lrate: f64, outputs: [f64; TARGET], targets: [f64; TARGET], act: &Activation<'a>) -> BackProps<COLS> {
        let BackProps(errors, gradients) = self.next.back_propagate(lrate, outputs, targets, act);
        let gradients = gradients.dot_multiply(&errors).map(&|x| x * lrate);

        self.weights = self.weights.add(&gradients.multiply(&self.data.transpose()));
        self.biases = self.biases.add(&gradients);

        let errors = self.weights.transpose().multiply(&errors);
        let gradients = self.data.map(&act.derivative);

        BackProps(errors, gradients)
    }
}


pub struct EndLayer<const TARGET: usize>();

impl <const COLS: usize> Layer<COLS, COLS> for EndLayer<COLS> {
    fn feed_forward<'a>(&mut self, feed: Matrix<COLS, 1>, _act: Activation<'a>) -> [f64; COLS] {
        feed.transpose().data[0]
    }

    fn back_propagate<'a>(&mut self, _lrate: f64, outputs: [f64; COLS], targets: [f64; COLS], act: &Activation<'a>) -> BackProps<COLS> {
        let parsed = Matrix::from([outputs]).transpose();
        let errors = Matrix::from([targets]).transpose().subtract(&parsed);
        let gradients = parsed.map(&act.derivative);
        BackProps (errors, gradients)
    }
}

pub struct BackProps<const COLS: usize>(Matrix<COLS, 1>, Matrix<COLS, 1>);
