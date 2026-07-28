

# 📌 Project Overview

Heart Disease Prediction is a Machine Learning project that predicts whether a patient is at risk of developing coronary heart disease (CHD) within the next 10 years.

The project uses **Logistic Regression**, a supervised machine learning classification algorithm, to estimate the probability of heart disease based on patient health information.

This project demonstrates the complete Machine Learning workflow, from data preprocessing and model training to evaluation, prediction, and deployment.

---

# 🎯 Business Problem

Heart disease is one of the leading causes of death worldwide.

Early prediction can help healthcare professionals identify high-risk patients and provide preventive treatment before serious complications occur.

The objective of this project is to build a classification model that predicts whether a patient is likely to develop heart disease using clinical and demographic features.

---

# 📊 Dataset

**Dataset:** Framingham Heart Study Dataset

The dataset contains patient information such as:

* Age
* Sex
* Smoking status
* Blood Pressure
* Cholesterol
* BMI
* Glucose
* Heart Rate
* Diabetes
* Hypertension Medication
* Previous Stroke
* Other clinical measurements

**Target Variable**

```text
TenYearCHD

0 → No Heart Disease

1 → Heart Disease
```

---

# 🤖 Machine Learning Workflow

```text
Business Problem
        │
        ▼
Data Loading
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Data Cleaning
        │
        ▼
Feature Selection
        │
        ▼
Train/Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Logistic Regression Training
        │
        ▼
Model Evaluation
        │
        ▼
Save Model
        │
        ▼
Prediction
        │
        ▼
Streamlit Deployment
```

---

# 🧠 Model Used

**Algorithm**

* Logistic Regression

**Library**

* Scikit-Learn

**Problem Type**

* Binary Classification

**Output**

* Probability of Heart Disease
* Predicted Class (0 or 1)

---

# 📈 Results

The model was evaluated using standard classification metrics, including:

* Accuracy
* ROC AUC Score
* Confusion Matrix
* Precision
* Recall
* F1-Score

The trained model was saved using Joblib for future predictions.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git

cd heart-disease-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

### Train the model

```bash
python train.py
```

### Make predictions

```bash
python predict.py
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

```text
heart-disease-prediction/

│── data/
│     └── framingham.csv
│
│── models/
│     ├── heart_disease_model.pkl
│     ├── scaler.pkl
│     └── feature_names.pkl
│
│── notebook.ipynb
│
│── train.py
│
│── predict.py
│
│── app.py
│
│── requirements.txt
│
│── README.md
│
│── .gitignore
│
└── images/
```

---

# 🚀 Future Improvements

Some possible improvements for future versions include:

* Train and compare multiple classification algorithms such as Random Forest, XGBoost, and Support Vector Machine (SVM).
* Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Improve feature engineering and feature selection.
* Add explainable AI techniques such as SHAP or LIME.
* Deploy the application using Streamlit Community Cloud.
* Build a REST API using FastAPI or Flask.
* Add Docker support for easier deployment.
* Collect more data to improve model performance.

---

Everything else—installation, project structure, and how to run—can follow the same format. This consistency makes your GitHub look organized and professional as your portfolio grows.
