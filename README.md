# ML Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models to predict Heart Disease based
on different properties. The goal is to build an end-to-end ML pipeline that includes model training, evaluation, and
deployment through an interactive Streamlit web application.

The assignment requires:

- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive web application for model comparison and prediction
- Provide option to test different model on the deployed app
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** Heart Disease Dataset from Kaggle

**Source:** [Kaggle - Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)

**Description:**
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It
contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14
of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease
and 1 = disease.
So This makes it a **_Binary Classification_** Problem.

**Features (13 input features):**

1. **age** - age in years
2. **sex** - 1=Male and 0=Female
3. **cp** - chest pain type (4 values)
4. **trestbps** - resting blood pressure(in mm Hg)
5. **chol** - serum cholestoral in mg/dl
6. **fbs** - fasting blood sugar > 120 mg/dl
7. **restecg** - resting electrocardiographic results (values 0,1,2)
8. **thalach** - maximum heart rate achieved
9. **exang** - exercise induced angina
10. **oldpeak** - ST depression induced by exercise relative to rest
11. **slope** - the slope of the peak exercise ST segment
12. **ca** - number of major vessels (0-3) colored by flourosopy
13. **thal** - 0 = normal; 1 = fixed defect; 2 = reversable defect

**Target**

1. **target** - 0 = no disease and 1 = disease

## Install Dependencies

```commandline
pip install -r requirements.txt
```

## Dataset Division

1. **Main Data** - heart.csv
2. **Train and Validation Data** - train.csv (90% of actual dataset)
3. **Test Data** - test.csv - (10% of actual dataset) (_Note - Also downloadable from Streamlit App_)

```commandline
 python prepare-dataset.py
```