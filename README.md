# Pima Indians Diabetes - EDA & Classification

Exploratory Data Analysis and Machine Learning models to predict diabetes diagnoses using the Pima Indians Diabetes dataset from Kaggle.

## Dataset Overview
The dataset contains medical diagnostic records for 768 individuals:
* **Pregnancies**: Number of times pregnant.
* **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test.
* **BloodPressure**: Diastolic blood pressure (mm Hg).
* **SkinThickness**: Triceps skin fold thickness (mm).
* **Insulin**: 2-Hour serum insulin (mu U/ml).
* **BMI**: Body mass index.
* **DiabetesPedigreeFunction**: Genetic predisposition score.
* **Age**: Age in years.
* **Outcome**: Target class variable (0 = No diabetes, 1 = Diabetes).

## Data Cleaning & Preprocessing
* **Missing Values**: Invalid `0` values in physiological metrics were replaced with `NaN`.
* **Imputation**: 
  * `Glucose` and `BloodPressure` were filled via a Truncated Normal Distribution based on feature means and standard deviations.
  * `SkinThickness`, `Insulin`, and `BMI` were filled using median values conditioned on the target `Outcome`.
* **Outliers**: Dropped values outside the 1.5 IQR threshold across continuous columns.
* **Feature Engineering**: Used `pd.cut()` to generate binned categories for `AgeGroup` (20–89) and `BMICategory` (Underweight to Obese III).
* **Transformations**: Implemented One-Hot Encoding for multi-value columns, Label Encoding for binary features, and standardized numerical data via `StandardScaler`.
* **Class Imbalance**: Applied SMOTE to oversample the minority class within the training split.

## Model Training & Performance
Data was partitioned into stratified Training (60%), Validation (20%), and Testing (20%) sets. Models were optimized using `GridSearchCV` with 5-fold cross-validation.

### Test Set Metrics

| Model | Test Accuracy | Test ROC-AUC |
| :--- | :---: | :---: |
| Logistic Regression | 76.62% | 0.8441 |
| Decision Tree | 83.76% | 0.8607 |
| **Random Forest** | **87.01%** | **0.9402** |

## Key Findings
* **Feature Importance**: `Insulin` was the top predictor by a large margin (importance score > 0.4), followed by `Glucose` and `SkinThickness`.
* **Granularity**: Raw continuous metrics provided significantly stronger predictive value to the tree-based models than the engineered categorical brackets.
