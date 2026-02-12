# ML Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models to predict whether a person's income exceeds $50K/yr based
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
An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.
This dataset contains 14 attributes extracted from the 1994 Census database.
So This makes it a **_Binary Classification_** Problem.

**Features (14 input features):**

1. **age** - continuous.
2. **workclass** - Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
3. **fnlwgt** - continuous.
4. **education** - Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
5. **education-num** - continuous.
6. **marital-status** - Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
7. **occupation** - Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
8. **relationship** - Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
9. **race** - White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
10. **sex** - Female, Male.
11. **capital-gain** - continuous.
12. **capital-loss** - continuous.
13. **hours-per-week** - continuous.
14. **native-country** - United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

**Target**

1. **income** - Income Class (<=50K, >50K)

## Install Dependencies

```commandline
pip install -r requirements.txt
```

## Dataset Division

1. **Main Data** - adult.csv
2. **Train and Validation Data** - train.csv (90% of actual dataset)
3. **Test Data** - test.csv - (10% of actual dataset) (_Note - Also downloadable from Streamlit App_)

```commandline
 python prepare_dataset.py
```