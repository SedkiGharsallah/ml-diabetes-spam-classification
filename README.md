# Imbalanced classification: diabetes and spam detection

Two end-to-end machine learning studies on tabular data, sharing a common set of
tested utility functions (`functions.py`). The focus is on **choosing the metric that
matches the real cost of each error**, rather than optimising accuracy.

| Study | Dataset | Classes | Key metric | Why |
|---|---|---|---|---|
| Diabetes | BRFSS 2015 health indicators (253,680 people, 21 features) | 86% / 14% | **Recall** | A missed diabetes case is worse than an extra medical test |
| Spam | UCI Spambase (4,601 e-mails, 57 word/character frequency features) | 61% / 39% | **Precision** | Blocking a legitimate e-mail is worse than letting a spam through |

## Pipeline

1. **Exploration:** class balance, missing values, correlation with the target.
2. **Preprocessing:** stratified train/test split, standardisation (fitted on the training set only).
3. **Class imbalance (diabetes):** SMOTE oversampling, applied to the training set only.
4. **Feature selection:** permutation importance implemented from scratch with NumPy,
   then selection of the top-k features (10 for diabetes, 20 for spam).
5. **Models:** Logistic Regression, Random Forest (with a hyperparameter grid search),
   K-Nearest Neighbors (K chosen by 10-fold stratified cross-validation),
   Gaussian Naive Bayes, and a PyTorch MLP (batch normalisation, dropout, early saving on validation loss).
6. **Comparison** on accuracy, precision, recall and F1, with confusion matrices.

## Results (test set)

Hyperparameters are tuned on a validation split carved out of the training set
(and, for KNN, by cross-validation with SMOTE applied inside each fold); the test set is used only once.

**Diabetes — sorted by recall**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.648 | 0.262 | **0.837** | 0.399 |
| Logistic Regression | 0.731 | 0.310 | 0.764 | 0.442 |
| Random Forest (tuned) | 0.742 | 0.313 | 0.714 | 0.436 |
| MLP (PyTorch) | 0.796 | 0.365 | 0.621 | **0.459** |
| Random Forest (default) | 0.792 | 0.356 | 0.613 | 0.451 |
| KNN (K=15) | 0.763 | 0.318 | 0.611 | 0.418 |

A model that always predicts "no diabetes" reaches 86% accuracy, which is why accuracy is not used to choose the model.
Naive Bayes catches 84% of diabetes cases, at the cost of many false alarms (precision 0.26).

**Spam — sorted by precision**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (default, 100 trees) | **0.947** | **0.949** | 0.915 | **0.931** |
| Random Forest (tuned) | 0.917 | 0.934 | 0.851 | 0.890 |
| Logistic Regression | 0.915 | 0.918 | 0.862 | 0.889 |
| MLP (PyTorch) | 0.921 | 0.901 | 0.898 | 0.899 |
| KNN (K=10) | 0.904 | 0.894 | 0.860 | 0.876 |
| Naive Bayes | 0.783 | 0.655 | 0.950 | 0.775 |

The default Random Forest gives the best precision. The tuned one is worse because the
search grid stops at 20 trees (see Limitations).

## Run it

```bash
pip install -r requirements.txt
```

Download the datasets and place them in this folder:
- `diabetes_binary_health_indicators_BRFSS2015.csv` — [Kaggle, Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- `spambase.data` — [UCI Machine Learning Repository, Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

Then open `diabetes_classification.ipynb` or `spam_classification.ipynb`.

Unit tests for the main utility functions (SMOTE balancing, permutation importance, feature selection):

```bash
pytest tests
```

## Limitations

- The Random Forest search grid is small (at most 20 trees) and the best depth lies on the
  edge of the grid; a wider search (more trees, smaller depths) would likely improve the tuned model.
- The decision threshold was left at 0.5. Tuning it (for example to reach a target recall)
  would be the natural next step, especially for the diabetes study.
- Permutation importance was computed with accuracy, while the models are compared on recall
  or precision; using the target metric would be more consistent.
