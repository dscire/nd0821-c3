# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model has been trained and deployed by Daniele Sciretti. The latest update of the model has been done in May 2025.

## Intended Use

This Random Forest Classifier predicts if an individual earns more or less than 50K based on a set of features as specified by the <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">income census dataset</a>.

## Training Data

The model has been trained on the census dataset described above, using 80% of the entries provided by the file. The training process uses GridSearchCV with a 5-fold cross validation.

## Evaluation Data

The model is evaluated on the 20% remaining data of the original dataset.

## Metrics

The model's performance is measured in terms of precision, recall and F1 score.
When measured on the test dataset, the deployed version of the model has produced the following results:
- precision = 0.8396
- recall = 0.4079
- F1 = 0.5490

## Ethical Considerations

The dataset contains information related to individuals from various races and genders. In this regards, there is a risk that the dataset may be biased in favor or against certain categories of individuals. A detailed fairness analysis of this dataset has not been carried out as part of this project.

## Caveats and Recommendations

The model has been chosen taking into consideration its reduced size, in order to reduce deployment time and operational costs. It should be considered as an exercise in API development and deployment, rather than a robust classification model.
