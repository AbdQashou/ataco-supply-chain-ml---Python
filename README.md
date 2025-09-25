# ataco-supply-chain-ml---Python
This repository contains a single Jupyter notebook that walks through an end-to-end applied machine-learning workflow on the DataCo Supply Chain dataset. The notebook covers:

Data loading, cleaning, and feature engineering

Categorical encoding and numeric scaling

Train/test splitting

Two predictive tasks:

Classification – predict Late_delivery_risk

Regression – predict Order Profit Per Order

Model training & evaluation (KNN for classification, Linear Regression for regression)

Basic confusion-matrix reporting and error metrics

1) Data preparation

Drops obvious non-essential fields (IDs, free text, duplicates, some location fields) and removes a few rows with missing key values.

Converts date columns to datetime, then derives calendar features: order_year, order_month, order_day, order_dayofweek.

Casts identifier-like numeric columns to strings so they don’t get treated as quantities.

Marks selected columns as categorical.

One-hot encodes selected categorical variables using pandas.get_dummies(..., drop_first=True) (classic one-hot encoding for ML). 
Pandas

Scales selected numeric features (the notebook shows both MinMaxScaler and StandardScaler usage on copies of the dataframe).

2) Feature/target definitions

Classification task

Target: Late_delivery_risk (binary)

Features: engineered numeric features + one-hot-encoded categoricals (with data-leakage-prone columns removed)

Regression task

Target: Order Profit Per Order

Features: similar cleaned/engineered set; highly collinear/leaky columns (e.g., profits derived from other totals) are dropped first

3) Split into train/test sets

Uses train_test_split to create holdout test sets (80/20 in the notebook). 
Scikit-learn
+1

4) Modeling

Classification (KNN):
The notebook trains K-Nearest Neighbors classifiers for k = 3..10, records metrics (Accuracy, Precision, Recall, F1), selects the best k by accuracy, and prints a confusion matrix for the best model. KNN is from scikit-learn’s neighbors module. 
Scikit-learn
+1

Regression (Linear Regression):
Trains an Ordinary Least Squares linear model and reports MAE, RMSE, and R² on the test set. 
Scikit-learn
+1

Note: There’s a short Keras Sequential model scaffold in the notebook, but no layers/training are defined yet. Treat that section as a placeholder if you plan to add a neural-network baseline later.

5) Evaluation

Classification: Accuracy, Precision, Recall, F1, and Confusion Matrix (printed in-notebook).

Regression: MAE, RMSE, R² (printed in-notebook).

#NOTES:
The notebook performs one-hot encoding on: Shipping Mode, Order Region, Market, Category Name, Department Name, Customer State (with drop_first=True to avoid the full dummy trap). 
Pandas

Both MinMaxScaler and StandardScaler appear; the main modeling cells scale the numeric features once prior to splitting/training. If you consolidate, keep a single scaler and apply fit on train / transform on test semantics (as shown in scikit-learn examples). 
Scikit-learn

KNN classification performs a simple grid over k = 3..10, tracks Accuracy/Precision/Recall/F1, then picks the k with the highest Accuracy and prints a confusion matrix. The estimator used is scikit-learn’s KNeighborsClassifier. 
Scikit-learn

Regression uses scikit-learn’s LinearRegression and reports MAE, RMSE, R².
