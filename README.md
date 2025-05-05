# Deep-Learning-Loan-Approval-

📌 Project Overview
This project is focused on predicting loan approval status (Approved or Default) using a deep learning model trained on a real-world financial dataset. The model leverages PyTorch, along with techniques such as feature engineering, regularization, class imbalance handling, and explainability (LIME and SHAP) to provide both accurate predictions and interpretable insights.

🎯 Objective
Predict whether a loan applicant is likely to repay or default.

Build a feedforward neural network capable of generalizing well.

Handle imbalanced classes, improve model robustness, and provide transparent explanations of decisions.

🧠 Dataset Description
Each record contains:

Applicant information: age, gender, education, income, employment experience, home ownership

Loan details: amount, interest rate, intent

Credit history: credit score, credit history length, previous defaults

Target: loan_status → 1 (Approved), 0 (Default)

🧰 Tools & Technologies
Language: Python 3.8+

Deep Learning: PyTorch

Preprocessing: Scikit-learn

Explainability: LIME, SHAP

Visualization: Matplotlib

Interactive UI: Streamlit (optional)

🔍 Key Features
✅ Data Preprocessing
Handled missing values, outliers, and type conversions

Applied label encoding and standard scaling

Created advanced engineered features (e.g. debt_to_income, income_stability, risk_score_est)

✅ Model Architecture
4-layer Feedforward Neural Network

LeakyReLU activations, BatchNorm, and Dropout for stability

Focal Loss and AdamW optimizer for robust training

✅ Training Strategy
Early stopping to prevent overfitting

Cosine Annealing LR scheduler

Class weighting and Focal Loss for handling imbalanced data

✅ Evaluation
Threshold-tuned prediction on test data

Metrics: Accuracy, Precision, Recall, F1 Score

Confusion matrix visualization

✅ Explainability
SHAP summary plots for global feature importance

LIME instance-wise explanations (text-based for Colab compatibility)

🚀 How to Use
Clone the repo or open in Colab

Upload the dataset loan_data.csv

Run preprocessing, training, and evaluation cells

Use the LIME section to explain predictions

Replace with new data (manual or batch) for live predictions

📝 Final Remarks
This project demonstrates not only a technically sound neural model but also a strong emphasis on interpretability, deployment-readiness, and ethical transparency — crucial for real-world financial applications.
