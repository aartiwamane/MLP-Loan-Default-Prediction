# MLP Loan Default Prediction

## Project Overview

This project implements a **Loan Default Prediction System** using a
**Multi-Layer Perceptron (MLP) Neural Network** with Scikit-learn.

The model predicts whether a loan applicant is likely to default based on
financial, employment, credit, and loan-related information.

The project also includes data preprocessing, feature scaling, model
evaluation, prediction on new loan applicants, and hyperparameter experiments.

---

## Objective

The main objective of this project is to build a neural network-based
classification model that can predict loan default based on applicant
information.

The project also demonstrates how different MLP hyperparameters affect
model performance.

---

##  Dataset

The dataset used in this project is:

`Loan_Default.csv`

### Features

- Age
- Income
- LoanAmount
- CreditScore
- EmploymentYears
- ExistingLoans
- MonthlyDebt
- LoanTerm
- PreviousDefault
- HomeOwnership

### Target Variable

`Default`

- `0` → No Default
- `1` → Default

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- MLPClassifier
- StandardScaler
- Train-Test Split

---

## Project Workflow

```text
Load Dataset
     ↓
Data Analysis
     ↓
Convert Categorical Data
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Create MLP Model
     ↓
Train Model
     ↓
Test Model
     ↓
Evaluate Model
     ↓
Plot Training Loss
     ↓
Test New Loan Applicants
     ↓
Hyperparameter Experiments
