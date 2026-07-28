# Load the data :
path = '/content/drive/MyDrive/Datasate/framingham.csv'
df = pd.read_csv(path)
df.head()

#EDA

df.shape
df.info

df.describe()
# Chack for messing values
df.isnull().sum()

# Clean the data
df_clean = df.dropna()
print (df_clean)

# Features selecting - Using all available features for maximum accuracy
y = df_clean['TenYearCHD']
X = df_clean.drop('TenYearCHD', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features to improve Logistic Regression accuracy
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Target variable: TenYearCHD")
print(f"Features used: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")
print("Data split and scaling complete.")

# Train the Logistic Regression model to learn the relationship
# between patient features and the risk of heart disease.
model = LogisticRegression()
model.fit(X_train, y_train)

# model prediction
y_pred = model.predict(X_test)
#Probability of each class
y_prob = model.predict_proba(X_test)

# Probability of positive class
positive_probability = y_prob[:,1]

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print (f'Accuracy of the model :{accuracy}')

print(classification_report(y_test, y_pred))
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance.sort_values(
    by="Coefficient",
    ascending=False)

# Define a new sample (you can change these values to test different scenarios)
new_sample = {
    'male': 1,
    'age': 45,
    'education': 2.0,
    'currentSmoker': 1,
    'cigsPerDay': 20.0,
    'BPMeds': 0.0,
    'prevalentStroke': 0,
    'prevalentHyp': 1,
    'diabetes': 0,
    'totChol': 220.0,
    'sysBP': 140.0,
    'diaBP': 90.0,
    'BMI': 28.5,
    'heartRate': 75.0,
    'glucose': 85.0
}

# 1. Convert to DataFrame (this ensures feature names and order match)
new_df = pd.DataFrame([new_sample])

# 2. Scale the data using the ALREADY FITTED scaler
new_sample_scaled = scaler.transform(new_df)

# 3. Make the prediction
prediction = model.predict(new_sample_scaled)
probability = model.predict_proba(new_sample_scaled)

print(f"Prediction for new sample: {'Risk of CHD' if prediction[0] == 1 else 'No Risk of CHD'}")
print(f"Probability of Heart Disease: {probability[0][1]*100:.2f}%")

# Fix: Convert dictionary to DataFrame before transforming
new_sample_df = pd.DataFrame([new_sample])

# Scale the data
new_sample_scaled = scaler.transform(new_sample_df)

print("--- Prediction Results ---")
print(f"Class Prediction: {prediction[0]}")
print(f"Status: {'Risk of CHD' if prediction[0] == 1 else 'No Risk of CHD'}")

print("\n--- Probabilities ---")
print(f"No Risk (0): {probability[0][0]:.4f}")
print(f"Risk (1): {probability[0][1]:.4f}")

import joblib
import os
from google.colab import files

# 1. Create the models directory
os.makedirs('models', exist_ok=True)

# 2. Save the components requested
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/features.pkl")

# 3. Create a copy specifically for downloading with the requested name
joblib.dump(model, 'heartdisease.pkl')

print("Artifacts saved to 'models/' folder.")

# 4. Trigger download
files.download('heartdisease.pkl')
  
