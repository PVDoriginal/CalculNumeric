use nalgebra::{DMatrix, DVector, Dyn, Matrix, Matrix1};
use rand::{Rng, rng};

use crate::utils::parse_csv;

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
    // Am impartit fisierul .pickle din laborator in 3 fisiere csv pentru fiecare graf ca sa le pot parsa fara numpy
    for graph in [
        parse_csv("graphs/g1.csv").unwrap(),
        parse_csv("graphs/g3.csv").unwrap(),
        parse_csv("graphs/g2.csv").unwrap(),
    ] {
        println!("Graph: {graph}");

        let eig = graph.symmetric_eigenvalues();
        println!("Eigenvalues: {eig}");
        println!("Max Clique: {}", eig.amax().floor() + 1.0);

        let mut unique_eig = eig.iter().map(|x| x.round() as i32).collect::<Vec<_>>();
        unique_eig.sort();
        unique_eig.dedup();

        println!("Is complete: {}", unique_eig.len() == 2);
        println!(
            "Is bipartite: {}",
            *unique_eig.iter().min().unwrap() == -1 * *unique_eig.iter().max().unwrap()
        );
    }
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
