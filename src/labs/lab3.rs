use core::f64;

use image::{ImageBuffer, ImageReader, Luma};
use nalgebra::Matrix3x4;
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

#[test]
fn ex2() {
    let img = ImageReader::open("image.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .to_luma16();

    let pixels: Vec<_> = img.as_raw().iter().map(|x| *x as f32).collect();

    let matrix = DMatrix::from_vec(img.width() as usize, img.height() as usize, pixels);

    for k in [
        img.height().min(img.width()) / 2,
        img.height().min(img.width()) / 5,
        img.height().min(img.width()) / 10,
    ] {
        let svd = matrix.clone().svd(true, true);

        let s: Vec<_> = svd.singular_values.iter().map(|x| *x).collect();
        let s = DMatrix::from_diagonal(&DVector::from_row_slice(&s[..k as usize]));

        let u = svd.u.unwrap();
        let u = u
            .clone()
            .remove_columns(k as usize, u.shape().1 - k as usize);

        let v = svd.v_t.unwrap();
        let v = v.clone().remove_rows(k as usize, v.shape().0 - k as usize);

        println!("{:?}", u.shape());
        println!("{:?}", s.shape());
        println!("{:?}", v.shape());

        let img = u * s * v;
        println!("{img}");

        let img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(
            img.shape().0 as u32,
            img.shape().1 as u32,
            img.iter().map(|x| *x as u16).collect(),
        )
        .unwrap();

        img.save(format!("image{k}.png")).unwrap();
    }
}
