use nalgebra::DMatrix;
use std::error::Error;
use std::fs;
use std::str::FromStr;

#[test]
fn read_g1() {
    println!("{}", parse_csv("graphs/g1.csv").unwrap());
}

// parses csv into Matrix
pub fn parse_csv(path: &str) -> Result<DMatrix<f32>, Box<dyn Error>> {
    let input = fs::read_to_string(path).unwrap();

    let mut data = Vec::new();
    let mut rows = 0;

    for line in input.lines() {
        rows += 1;
        for datum in line.split_terminator(",") {
            data.push(f32::from_str(datum.trim())?);
        }
    }
    let cols = data.len() / rows;

    Ok(DMatrix::from_row_slice(rows, cols, &data[..]))
}
