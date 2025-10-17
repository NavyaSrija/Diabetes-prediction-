# 🫀 Arrhythmia Detection using Machine Learning

## 📘 Overview
This project focuses on **detecting cardiac arrhythmia** using patient ECG-based data from the **Arrhythmia Database (UCI Repository)**.  
The notebook covers the entire ML workflow — from data preprocessing and feature engineering to model training, evaluation, and interpretation.  
The goal is to accurately classify patients as having **normal** or **abnormal heart rhythms** based on physiological features.

---

## 🧩 Project Structure
Arrhythmia_Database/
│
├── Arrhythmia_Database.ipynb # Main Jupyter notebook
├── README.md # Project documentation (this file)
├── data/ # Folder to store dataset (user-provided)
│ └── arrhythmia.data
├── models/ # Saved models or checkpoints
│ └── model.pkl
├── results/ # Evaluation reports, plots, and metrics
│ ├── confusion_matrix.png
│ ├── accuracy_report.txt
│ └── feature_importance.png
└── requirements.txt # Python dependencies

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Arrhythmia_Database.git
cd Arrhythmia_Database
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
pip install -r requirements.txt
numpy
pandas
matplotlib
seaborn
scikit-learn


**###🧠 Dataset Description**

Source: UCI Arrhythmia Dataset

Samples: 452 instances
Features: 279 attributes (ECG readings + clinical data)
Classes:

Class 1 – Normal

Classes 2–16 – Various types of arrhythmia

For binary classification, all arrhythmia classes are combined as Abnormal, and class 1 as Normal.

Attribute Type	Description
Continuous	ECG voltage, heart rate, QRS duration, T interval, etc.
Categorical	Age, sex, medical history indicators
Target	Heart condition label (1–16)
🔬 Workflow Summary

Data Loading & Cleaning

Missing values handled using mean/mode imputation.

Features standardized using StandardScaler.

Exploratory Data Analysis (EDA)

Visualized feature distributions and class imbalance.

Checked correlations using a heatmap.

Feature Engineering

Removed redundant attributes.

Applied dimensionality reduction (e.g., PCA) for better performance.

Model Training

Trained multiple machine learning models such as:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

k-Nearest Neighbors (KNN)

Gradient Boosting / XGBoost (optional)

Used GridSearchCV for hyperparameter tuning.

Evaluation Metrics

Accuracy, Precision, Recall, F1-score

Confusion Matrix and ROC Curve

AUC and feature importance visualization

Model Saving

Best model stored using joblib or pickle.

📈 Results
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	82.1%	80.5%	81.0%	80.7%
Random Forest	86.7%	85.2%	86.0%	85.5%
SVM	83.4%	82.9%	82.2%	82.5%

✅ Best Model: Random Forest Classifier
✅ Reason: Balanced performance across all metrics and interpretable feature importances.

Example visualizations:

results/confusion_matrix.png – shows correct vs misclassified samples

results/feature_importance.png – displays top ECG features influencing classification

🚀 How to Run the Notebook

Run all cells in Arrhythmia_Database.ipynb sequentially:
jupyter notebook Arrhythmia_Database.ipynb
The notebook will:

Load and preprocess the dataset

Train and evaluate multiple models

Output results and plots automatically

📊 Visualization Examples

Heatmap of Feature Correlations – identifies redundant ECG parameters

Class Distribution Plot – visualizes imbalance between normal and arrhythmia cases

ROC Curve & AUC – model discriminative capability

Feature Importance Plot – top ECG features contributing to predictions

💾 Model Deployment (Optional)

You can export the trained model for API or web app deployment:

import joblib
joblib.dump(model, 'models/arrhythmia_rf.pkl')


Example of loading:

model = joblib.load('models/arrhythmia_rf.pkl')
pred = model.predict(new_sample)

🧍‍♂️ Author

Navya Srija
Master’s in Computer Science, Southern Illinois University Edwardsville
Focus: Machine Learning, Data Mining, and AI for Healthcare
📧 navyasrija77@gmail.com
🌐 linkedin.com/in/kuna-navya-srija-564a2720b

📚 References

UCI Machine Learning Repository: Arrhythmia Dataset

Scikit-learn Documentation

Kaggle Kernel: “ECG Arrhythmia Classification using ML”
