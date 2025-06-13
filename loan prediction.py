import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load the dataset
df = pd.read_csv('loan_data_set.csv')

# Drop Loan_ID column as it is not useful for prediction
df = df.drop('Loan_ID', axis=1)

# Impute missing values
# Numerical columns: LoanAmount, Loan_Amount_Term, Credit_History
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    # Fix for FutureWarning: chained assignment using inplace method
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: Gender, Married, Dependents, Self_Employed
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    # Fix for FutureWarning: chained assignment using inplace method
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode 'Dependents' column manually due to its ordinal nature
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Separate features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Convert target variable 'Loan_Status' to numerical (Y=1, N=0)
le = LabelEncoder()
y = le.fit_transform(y)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Apply One-Hot Encoding to categorical features
# Fix for FutureWarning: `sparse` was renamed to `sparse_output`
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)

# Combine encoded categorical features with numerical features
X_processed_for_training = pd.concat([X[numerical_features], encoded_df], axis=1)

# Store the exact column names and order after full preprocessing on training data
# This is crucial for matching new data columns for prediction
fitted_feature_names = X_processed_for_training.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed_for_training, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
# Note: Ensure numerical_features are passed correctly to scaler
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
# Make predictions on test set
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# --- New Data Prediction Section ---
print("\n--- New Data Prediction ---")

# Example new data point
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

# Convert to DataFrame
new_df = pd.DataFrame([new_applicant_data])

# --- Preprocessing for New Data (MUST mirror training preprocessing) ---

# 1. Handle missing values (using medians/modes from original training data)
# Use original df for median/mode reference for robustness, though not strictly needed for this example
# because new_applicant_data has no missing values.
for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    if col in new_df.columns:
        new_df[col] = new_df[col].fillna(df[col].median())

for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    if col in new_df.columns:
        new_df[col] = new_df[col].fillna(df[col].mode()[0])

# 2. Encode 'Dependents'
new_df['Dependents'] = new_df['Dependents'].replace('3+', 3).astype(int)

# 3. Apply One-Hot Encoding to new categorical features using the *fitted* encoder
new_encoded_features = encoder.transform(new_df[categorical_features])
new_encoded_df = pd.DataFrame(new_encoded_features, columns=encoder.get_feature_names_out(categorical_features), index=new_df.index)

# Combine encoded categorical features with numerical features for the new data
# This initial concat might still have missing columns or different order
new_X_combined = pd.concat([new_df[numerical_features], new_encoded_df], axis=1)

# CRUCIAL STEP: Ensure columns match the training data's fitted_feature_names
# Reindex to ensure all columns from training are present, filling new ones with 0,
# and dropping any extra columns not seen during training.
new_X_processed = new_X_combined.reindex(columns=fitted_feature_names, fill_value=0)

# 4. Scale numerical features using the *fitted* scaler
new_X_processed[numerical_features] = scaler.transform(new_X_processed[numerical_features])

# ### Step 4: Make a Prediction
predicted_status_encoded = knn.predict(new_X_processed)

# Convert the numerical prediction back to original labels (Y or N)
predicted_status = le.inverse_transform(predicted_status_encoded)

print(f"The predicted loan status for the new applicant is: {predicted_status[0]}")