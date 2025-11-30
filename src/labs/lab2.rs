use nalgebra::{DMatrix, DVector, Dyn, Matrix, Matrix1};
use rand::{Rng, rng};

#[test]
fn ex1() {
    let mut rng = rng();

    // 6 x 6 random matrix
    let n = 6;
    let mut mat_a = Matrix::<f64, Dyn, Dyn, _>::from_fn(n, n, |_, _| rng.random::<f64>());

    mat_a = mat_a.clone() + mat_a.clone().transpose();

    println!("A: {}", mat_a);

    let eig = eigenvector(&mat_a);

    println!("Calculated Eigenvector: {eig}");
    println!(
        "Nalgebra Eigenvectors: {}:",
        mat_a.symmetric_eigen().eigenvectors.column(0)
    );
}

#[test]
fn ex2() {
    todo!()
}

const TOLERANCE: f64 = 0.000001;
const MAX_ITER: u32 = 1000;
fn eigenvector(mat_a: &DMatrix<f64>) -> DVector<f64> {
    let mut rng = rng();

    let mut y = DVector::from_fn(mat_a.shape().clone().0, |_, _| rng.random::<f64>()).normalize();

    for _ in 0..MAX_ITER {
        let z = (mat_a.clone() * y.clone()).normalize();
        let err = (Matrix1::new(1.0) - (z.transpose() * y.clone()).abs()).abs()[0];

        y = z;

        if err <= TOLERANCE {
            break;
        }
    }

    y
}
