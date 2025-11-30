use core::f64;

use nalgebra::{DMatrix, DVector, Matrix1};
use rand::{Rng, rng};
use rand_distr::Distribution;
use rand_distr::Normal;

#[test]
fn ex1() {
    let mut rng = rng();

    // a

    let m = 10;
    let r = 4;

    let mut mat_a = DMatrix::from_fn(m, r, |_, _| rng.random::<f64>());

    println!("A: {mat_a}");
    println!("Rank: {}", mat_a.rank(f64::EPSILON));

    // b

    for c in 0..4 {
        mat_a = mat_a.clone().insert_column(mat_a.shape().1, 1.);

        let copy = mat_a.clone();
        let column = copy.column(c) * Matrix1::new((c + 2) as f64);

        mat_a.set_column(mat_a.shape().1 - 1, &column);
    }

    println!("A: {mat_a}");

    // c
    mat_a += DMatrix::from_fn(m, r + 4, |_, _| {
        Normal::new(0.0, 0.2).unwrap().sample(&mut rng)
    });

    println!("A: {mat_a}");
    println!("Rank: {}", mat_a.rank(f64::EPSILON));

    // d
    let svd = mat_a.svd(true, true);
    println!("Singular values: {}", svd.singular_values);
}

fn ex2() {
    todo!()
}
