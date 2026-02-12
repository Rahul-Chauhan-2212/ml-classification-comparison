# ML Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models to predict whether a person's
income exceeds $50K/yr based
on census data. The goal is to build an end-to-end ML pipeline that includes model training, evaluation, and
deployment through an interactive Streamlit web application.

The assignment requires:

- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive web application for model comparison and prediction
- Provide option to test different model on the deployed app
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** Adult Income Dataset from Kaggle

**Source:** [Kaggle - Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

**Description:**
An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education
level, age, gender, occupation, and etc.
This dataset contains 14 attributes extracted from the 1994 Census database.
So This makes it a **_Binary Classification_** Problem.

**Features (14 input features):**

1. **age** - continuous.
2. **workclass** - Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay,
   Never-worked.
3. **fnlwgt** - continuous.
4. **education** - Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
   Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
5. **education-num** - continuous.
6. **marital-status** - Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,
   Married-AF-spouse.
7. **occupation** - Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
   Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
   Protective-serv, Armed-Forces.
8. **relationship** - Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
9. **race** - White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
10. **sex** - Female, Male.
11. **capital-gain** - continuous.
12. **capital-loss** - continuous.
13. **hours-per-week** - continuous.
14. **native-country** - United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

**Target**

1. **income** - Income Class (<=50K, >50K)

### Install Dependencies

```commandline
pip install -r requirements.txt
```

### Dataset Division

1. **Main Data** - adult.csv
2. **Train and Validation Data** - train.csv (90% of actual dataset)
3. **Test Data** - test.csv - (10% of actual dataset) (_Note - Also downloadable from Streamlit App_)

```commandline
 python prepare_dataset.py
```

## Models Used

### Model Performance Comparison

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|---------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression | 0.8551   | 0.9078 | 0.7488    | 0.6264 | 0.6821   | 0.5932 |
| Decision Tree       | 0.8672   | 0.9089 | 0.8034    | 0.6155 | 0.6970   | 0.6228 |
| KNN                 | 0.8342   | 0.8744 | 0.6855    | 0.6135 | 0.6475   | 0.5410 |
| Naive Bayes         | 0.6485   | 0.8510 | 0.4065    | 0.9049 | 0.5610   | 0.4072 |
| Random Forest       | 0.8625   | 0.9184 | 0.8082    | 0.5847 | 0.6786   | 0.6065 |
| XGBoost             | 0.8763   | 0.9312 | 0.7983    | 0.6710 | 0.7291   | 0.6539 |

## Observations about Model Performance

| ML Model Name           | Observation about model performance                                                                                                                                    |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | Strong baseline model with **high AUC (0.9078)** and balanced metrics. Good generalization, but recall is moderate, indicating some false negatives.                   |
| **Decision Tree**       | Slightly better accuracy than logistic regression. Captures non-linear patterns well, but prone to overfitting without pruning. Precision is high, recall moderate.    |
| **KNN**                 | Decent performance but lower accuracy/AUC compared to tree-based ensembles. Sensitive to feature scaling and choice of K. Works reasonably but not optimal here.       |
| **Naive Bayes**         | Very **high recall (0.9049)** but poor precision. Tends to over-predict the positive class. Useful when minimizing false negatives is critical.                        |
| **Random Forest**       | Robust and stable with **high precision (0.8082)** and strong AUC. Handles feature interactions well. Slight recall drop suggests conservative predictions.            |
| **XGBoost**             | **Best overall performer** with highest accuracy, AUC, F1, and MCC. Excellent balance between precision and recall. Clearly the most effective model for this dataset. |

### Summary

* **Best Overall Model:** XGBoost
* **Most Conservative (High Precision):** Random Forest
* **Most Sensitive (High Recall):** Naive Bayes
* **Best Linear Baseline:** Logistic Regression


