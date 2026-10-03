# Titanic Survival Prediction — Tabular Classification

## Overview

This project is an end-to-end machine learning project that predicts whether a passenger survived the Titanic disaster.

The project was completed as part of the LeetVerse AI/ML probation task.

## Problem Statement

Given passenger information such as passenger class, sex, age, number of siblings/spouses, parents/children, fare, and embarkation point, predict whether the passenger survived.

This is a **binary classification** problem.

- `0` → Did not survive
- `1` → Survived

## Dataset

The project uses the Titanic dataset available through the Seaborn library.

The dataset contains **891 passenger records** and **15 columns**.

### Important Features

- `pclass` — Passenger class
- `sex` — Passenger sex
- `age` — Passenger age
- `sibsp` — Number of siblings/spouses aboard
- `parch` — Number of parents/children aboard
- `fare` — Passenger fare
- `embarked` — Port of embarkation

### Target

`survived`

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset and identify missing values and patterns.

The dataset contained missing values mainly in:

- `age`
- `embarked`
- `deck`

A count plot was used to examine the distribution of survival outcomes.

Another visualization compared survival counts by sex.

## Data Preprocessing

The following preprocessing steps were performed:

1. Selected relevant features.
2. Filled missing age values using the median age.
3. Filled missing embarkation values using the most frequent value.
4. Split the data into training and testing sets.
5. Used One-Hot Encoding for categorical features such as `sex` and `embarked`.

The dataset was split into:

- Training data: **712 samples**
- Testing data: **179 samples**

## Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Missing Value Handling
   ↓
Train/Test Split
   ↓
Categorical Feature Encoding
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Overfitting Analysis
   ↓
Error Analysis
```
## Models Used

### 1. Logistic Regression

Logistic Regression was used as a classification model to predict the probability of passenger survival.

### 2. Random Forest

Random Forest was used as an ensemble classification model consisting of multiple decision trees.

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.45% | 79.31% | 66.67% | 72.44% |
| Random Forest | 82.12% | 81.36% | 69.57% | 75.00% |

Random Forest produced higher values on all four test metrics.

However, Random Forest had a training accuracy of **98.17%** compared with a testing accuracy of **82.12%**, showing a substantial train-test gap.

Logistic Regression had a training accuracy of **80.76%** and testing accuracy of **80.45%**, showing much more consistent performance between training and testing data.

## Overfitting Analysis

Training and testing accuracies were compared to identify possible overfitting.

### Logistic Regression

- Training Accuracy: **80.76%**
- Testing Accuracy: **80.45%**
- Train-Test Gap: **approximately 0.31 percentage points**

The small difference indicates relatively consistent performance between training and testing data.

### Random Forest

- Training Accuracy: **98.17%**
- Testing Accuracy: **82.12%**
- Train-Test Gap: **approximately 15.89 percentage points**

The larger difference indicates evidence of overfitting.

## Confusion Matrix

A confusion matrix was used to examine the Random Forest predictions.

| | Predicted Did Not Survive | Predicted Survived |
|---|---:|---:|
| Actual Did Not Survive | 99 | 11 |
| Actual Survived | 21 | 48 |

This helps identify both correct and incorrect classifications made by the model.

## Error Analysis

One incorrect prediction was examined in detail.

For passenger **553**:

- Actual outcome: **Survived**
- Predicted outcome: **Did not survive**
- Age: **22**
- Sex: **Male**
- Passenger class: **3**
- Fare: **7.225**

This demonstrates that a machine learning model can make individual errors even when its overall test performance is good.

The available features do not establish the exact reason why the model made this particular prediction.

## Model Comparison and Conclusion

Random Forest achieved higher test-set Accuracy, Precision, Recall, and F1 Score than Logistic Regression.

However, Random Forest also showed substantially more evidence of overfitting because of its large difference between training and testing accuracy.

Logistic Regression produced slightly lower test metrics but had a much smaller train-test gap and provides a simpler model.

For this project, I would keep **Logistic Regression** because it provides a much smaller train-test gap and a simpler model, while its test performance is only slightly lower than Random Forest.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Key Learnings

- Learned the basic workflow of an end-to-end machine learning project.
- Learned exploratory data analysis using Pandas, Matplotlib, and Seaborn.
- Learned how to handle missing values.
- Learned categorical feature encoding using One-Hot Encoding.
- Learned the difference between training and testing data.
- Learned how Logistic Regression can be used for classification.
- Learned how Random Forest works using multiple decision trees.
- Learned Accuracy, Precision, Recall, and F1 Score.
- Learned how to identify overfitting using the training-testing performance gap.
- Learned how to perform basic error analysis.

## Limitation

This project uses a single train-test split and a limited set of passenger features.

In a more rigorous implementation, preprocessing could be placed inside a machine-learning pipeline so that preprocessing parameters are learned only from the training data.

## How to Run

The project can be opened and executed using Google Colab.

1. Open the notebook.
2. Open it in Google Colab.
3. Run the cells from top to bottom.
4. The dataset is loaded directly using the Seaborn library.

## Project Structure

```text
Titanic-Classification/
│
├── Leetverse_probation_task.ipynb
└── README.md
```
## Project Goal

The goal of this project was not only to build a classification model, but also to understand the complete machine learning workflow from data exploration and preprocessing to model evaluation, overfitting analysis, and error analysis.
