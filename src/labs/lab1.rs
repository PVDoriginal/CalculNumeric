use std::path::Path;

use nalgebra::{DMatrix, DVector};
use plotters::{
    prelude::*,
    series::{LineSeries, PointSeries},
};
use rand::{Rng, rng};

use lstsq::lstsq;
use rust_decimal::{Decimal, dec};

#[test]
fn ex1() {
    let f = |x: Decimal| {
        if x < dec!(0.5) {
            dec!(2) * x
        } else {
            dec!(2) * x - dec!(1)
        }
    };
    let mut n = dec!(0.1);

    for _ in 0..60 {
        println!("{n}");
        n = f(n);
    }
}

#[test]
fn ex2() {
    // Solve example matrix
    let matrix = [[2, 4, -2], [4, 9, -3], [-2, -3, 7]];
    let b = [2, 8, 10];

    solve_matrix(matrix, b);

    // Solve random matrix
    let mut rng = rng();

    let mut matrix = [[0; 6]; 6];
    let mut b = [0; 6];

    b = b.map(|_| rng.random_range(1..100));
    matrix = matrix.map(|m| m.map(|_| rng.random_range(1..100)));

    solve_matrix(matrix, b);
}

fn solve_matrix<const N: usize>(mut matrix: [[i32; N]; N], mut b: [i32; N]) {
    println!("A: {matrix:?}");
    println!("b: {b:?}");

    for k in 0..N - 1 {
        for i in k + 1..N {
            matrix[i][k] = -matrix[i][k] / matrix[k][k];

            for j in k + 1..N {
                matrix[i][j] = matrix[i][j] + matrix[k][j] * matrix[i][k];
            }

            b[i] = b[i] + b[k] * matrix[i][k];
        }
    }

    let mut x = b;

    for i in (0..N).rev() {
        for j in i + 1..N {
            x[i] = x[i] - matrix[i][j] * x[j];
        }
        x[i] = x[i] / matrix[i][i];
    }

    println!("X: {x:?}");
}

#[derive(serde::Deserialize, Clone)]
struct Record(pub f64, pub f64);

#[test]
fn ex3() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(Path::new("regresie.csv"))?;

    let records = reader.deserialize::<Record>();

    let n_points = csv::Reader::from_path(Path::new("regresie.csv"))?
        .deserialize::<Record>()
        .count();

    let mut a = DMatrix::from_element(n_points, 2, 1.);
    let mut b = DVector::from_element(n_points, 0.);

    let mut points: Vec<(f32, f32)> = Vec::default();

    for (i, point) in records.enumerate() {
        let point = point?;

        a[(i, 0)] = point.0;
        b[i] = point.1;

        points.push((point.0 as f32, point.1 as f32));
    }

    let line = lstsq(&a, &b.to_owned(), 1e-14)?;

    let root = BitMapBackend::new("lab1_ex3_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2f32..1f32, -2f32..2f32)?;

    chart.draw_series(PointSeries::of_element(points, 3, &RED, &|c, s, st| {
        return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
    }))?;

    chart.draw_series(LineSeries::new(
        (-200..=100)
            .map(|x| x as f32 / 100.0)
            .map(|x| (x, line.solution[0] as f32 * x + line.solution[1] as f32)),
        &RED,
    ))?;

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()?;

    root.present()?;

    Ok(())
}
