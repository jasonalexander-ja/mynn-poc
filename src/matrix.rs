use rand::{thread_rng, Rng};
use super::Float;


#[derive(Clone)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
	pub data: [[Float; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
	pub fn zeros() -> Matrix<ROWS, COLS> {
		Matrix {
			data: [[0.0; COLS]; ROWS]
		}
	}

	pub fn random() -> Matrix<ROWS, COLS> {
		let mut rng = thread_rng();
		let mut data = [[0.0; COLS]; ROWS];

		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = rng.gen::<Float>() * 2.0 - 1.0;
			}
		}

		Matrix {
			data
		}
	}

	pub fn multiply<const OTHER_COLS: usize>(&self, other: &Matrix<COLS, OTHER_COLS>) -> Matrix<ROWS, OTHER_COLS> {

		let mut res = Matrix::<ROWS, OTHER_COLS>::zeros();

		for i in 0..ROWS {
			for j in 0..OTHER_COLS {
				let mut sum = 0.0;
				for k in 0..COLS {
					sum += self.data[i][k] * other.data[k][j];
				}

				res.data[i][j] = sum;
			}
		}

		res
	}

	
	pub fn add(&self, other: &Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = self.data[row][col] + other.data[row][col];
			}
		}

		Matrix {
			data
		}
	}

	pub fn dot_multiply(&self, other: &Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = self.data[row][col] * other.data[row][col];
			}
		}

		Matrix {
			data
		}
	}

	
	pub fn subtract(&self, other: &Matrix<ROWS, COLS>) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = self.data[row][col] - other.data[row][col];
			}
		}

		Matrix {
			data
		}
	}

	pub fn map(&self, function: &dyn Fn(Float) -> Float) -> Matrix<ROWS, COLS> {

		let mut data = [[0.0; COLS]; ROWS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[row][col] = function(self.data[row][col]);
			}
		}

		Matrix {
			data
		}
	}

	pub fn from(data: [[Float; COLS]; ROWS]) -> Matrix<ROWS, COLS> {
		Matrix {
			data
		}
	}

	pub fn transpose(&self) -> Matrix<COLS, ROWS> {
		let mut data = [[0.0; ROWS]; COLS];
		for row in 0..ROWS {
			for col in 0..COLS {
				data[col][row] = self.data[row][col];
			}
		}
		Matrix {
			data
		}
	}
}
