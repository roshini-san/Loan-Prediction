Sure! Here's the cleaned-up, emoji-free version of your **README.md**:

---

# Loan Prediction System using KNN

This project predicts whether a loan should be approved or not, based on applicant details. It uses a clean preprocessing pipeline and a K-Nearest Neighbors (KNN) model to classify loan applications as either **approved (Y)** or **rejected (N)**.

## Files

* `loan_prediction.py`: Python script containing the full workflow — data cleaning, preprocessing, training, evaluation, and prediction.
* `loan_data_set.csv`: Dataset containing past loan applications with features like income, education, credit history, etc.

## Features

* End-to-end data processing pipeline:

  * Handling missing values
  * Encoding categorical variables (label + one-hot encoding)
  * Scaling numerical features
* Model training using `KNeighborsClassifier`
* Evaluation metrics: Accuracy, Precision, Recall, F1-Score
* A separate section for making predictions on **new applicant data** with proper feature alignment and preprocessing.

## Tech Stack

* Python
* scikit-learn
* pandas
* numpy

## Model Workflow

1. **Load & Clean Data**:

   * Drop irrelevant columns (like `Loan_ID`)
   * Fill missing values (median for numeric, mode for categorical)
2. **Encode Data**:

   * `Dependents` is label encoded (with ‘3+’ mapped to 3)
   * Categorical variables are one-hot encoded
3. **Split & Scale**:

   * 80-20 train-test split
   * Apply standard scaling to numerical features
4. **Train Model**:

   * Trains a KNN model with `n_neighbors=5`
5. **Evaluate**:

   * Outputs accuracy, precision, recall, and F1-score
6. **Predict New Input**:

   * Preprocesses new data using the same steps and predicts loan approval status

## Sample Prediction

```python
new_applicant_data = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '2',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 6000,
    'CoapplicantIncome': 1000.0,
    'LoanAmount': 150.0,
    'Loan_Amount_Term': 360.0,
    'Credit_History': 1.0,
    'Property_Area': 'Semiurban'
}
```

Output:

```
The predicted loan status for the new applicant is: Y
```

## Sample Evaluation Output

```
--- Model Evaluation ---
Accuracy: 0.7892
Precision: 0.8049
Recall: 0.8889
F1-Score: 0.8444
```

## Notes

* Make sure the new data has no missing values or use the original training data’s median/mode to fill them.
* Ensure the new data's features match the training feature order and structure before prediction.
* The model may be improved by trying other classifiers like Logistic Regression, Random Forest, or XGBoost.

## How to Run

```bash
pip install pandas scikit-learn numpy
python loan_prediction.py
```

