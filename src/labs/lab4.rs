use linfa::{
    MultiClassModel,
    prelude::{Pr, ToConfusionMatrix},
    traits::{Fit, Predict},
};
use linfa_svm::Svm;

#[test]
fn ex1() {
    let (train, test) = linfa_datasets::iris().split_with_ratio(0.75);

    // linear kernel
    {
        let params = Svm::<_, Pr>::params().linear_kernel();

        let model = train
            .one_vs_all()
            .unwrap()
            .into_iter()
            .map(|(l, x)| (l, params.fit(&x).unwrap()))
            .collect::<MultiClassModel<_, _>>();

        let pred = model.predict(&test);
        let cm = pred.confusion_matrix(&test).unwrap();

        println!("linear kernel results:");
        println!("{cm:?}");
        println!("accuracy: {}", cm.accuracy());
        println!();
    }

    // rbf kernel
    {
        let params = Svm::<_, Pr>::params().gaussian_kernel(30.0);

        let model = train
            .one_vs_all()
            .unwrap()
            .into_iter()
            .map(|(l, x)| (l, params.fit(&x).unwrap()))
            .collect::<MultiClassModel<_, _>>();

        let pred = model.predict(&test);
        let cm = pred.confusion_matrix(&test).unwrap();

        println!("rbf kernel results:");
        println!("{cm:?}");
        println!("accuracy: {}", cm.accuracy());
        println!();
    }
}
