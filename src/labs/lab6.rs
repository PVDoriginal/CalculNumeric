use std::f64::consts::{E, PI};

use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, IntoDrawingArea, IntoLinspace, IntoSegmentedCoord, Rectangle},
    series::Histogram,
    style::{BLUE, Color, GREEN, WHITE},
};
use rand::rng;
use rand_distr::Distribution;
use rand_distr::Normal;

use probability::{
    distribution::{Gaussian, Uniform},
    prelude::Continuous,
};

#[test]
fn ex1() -> Result<(), Box<dyn std::error::Error>> {
    let mean = 90.0;
    let dev = 10.0;

    let mut rng = rng();

    let root = BitMapBackend::new("lab6_ex1_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..180.0, 0.0..10.0)?
        .set_secondary_coord(
            (-0.0..180.0).step(0.1).use_round().into_segmented(),
            0u32..10u32,
        );

    let normal = Normal::new(mean, dev).unwrap();

    let points = (1..600).map(|_| normal.sample(&mut rng));

    let histogram = Histogram::vertical(chart.borrow_secondary())
        .style(BLUE.filled())
        .margin(10)
        .data(points.map(|x| (x, 1)));

    chart
        .draw_secondary_series(histogram)?
        .label("Observed")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], GREEN.filled()));

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()?;

    root.present()?;

    Ok(())
}

fn verosimility(mean: f64, dev: f64, x: f64) -> f64 {
    (1.0 / (2.0 * PI * dev * dev).sqrt()) * E.powf((-(x - mean) * (x - mean)) / (2.0 * dev * dev))
}

#[test]
fn ex2() {
    let mean = 90.0;
    let dev = 10.0;
    let x = 82.0;

    let normal = Gaussian::new(mean, dev).density(x);

    println!("density: {normal}");
    println!("calculated verosimility: {}", verosimility(mean, dev, x));
}

fn data() -> Vec<f64> {
    vec![
        82.0, 106.0, 120.0, 68.0, 83.0, 89.0, 130.0, 92.0, 99.0, 89.0,
    ]
}

fn total_verosimility(mean: f64, dev: f64) -> f64 {
    let mut res = 1.0;
    for x in data() {
        res *= verosimility(mean, dev, x);
    }
    res
}

#[test]
fn ex3() {
    println!("total verosimility: {}", total_verosimility(90.0, 10.0));
}

fn apriori(mean: f64, dev: f64) -> f64 {
    let norm = Gaussian::new(100.0, 50.0);
    let unif = Uniform::new(1.0, 70.0);

    norm.density(mean) * unif.density(dev)
}

#[test]
fn ex4() {
    println!("apriori: {}", apriori(90.0, 10.0));
}

fn aposteriori(mean: f64, dev: f64) -> f64 {
    apriori(mean, dev) * total_verosimility(mean, dev)
}

#[test]
fn ex5() {
    println!("aposteriori: {}", aposteriori(90.0, 10.0));
}

#[test]
fn ex6() {
    let means = vec![70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0];
    let devs = vec![5.0, 10.0, 15.0, 20.0];

    let mut best = (0.0, 0.0, -100.0);

    for mean in means {
        for &dev in &devs {
            let res = aposteriori(mean, dev);
            println!("mean: {mean}, dev: {dev}, = aposteriori: {res}");

            if res > best.2 {
                best = (mean, dev, res);
            }
        }
    }

    println!();
    println!("Best value:");
    println!(
        "mean: {}, dev: {}, = aposteriori: {}",
        best.0, best.1, best.2
    );
}
