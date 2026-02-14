Bank Transaction Fraud Detection System

Used dataset link :https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection

I built a machine learning system to detect fraudulent bank transactions using multiple classification algorithms. The objective of this project was to compare different models and determine which one performs best, especially when dealing with an imbalanced dataset.

In this project, I implemented three machine learning models: Logistic Regression, Random Forest, and XGBoost. Since fraud detection datasets are usually highly imbalanced, I handled the class imbalance problem using SMOTE (Synthetic Minority Over-sampling Technique).

To evaluate model performance properly, I used multiple metrics including ROC-AUC Score, F1-Score, Confusion Matrix, and Classification Reports. I also generated visualizations such as ROC curves, confusion matrices, and model comparison charts to clearly analyze and compare results.

Setup Instructions

First, install the required dependencies using:
pip install -r requirements.txt

Then run the script using:
python fraud_detection.py

On the first run, the dataset is automatically downloaded from Kaggle, so no manual download is required.

Optional: Using a Local Dataset

If you prefer using a local CSV file instead of automatic download, you can download the dataset from Kaggle, place it inside the project directory, and modify the code like this:

detector = FraudDetectionModel(data_path="your_file.csv")

How the Pipeline Works

When the script runs, it performs the following steps:

1. Loads and analyzes the dataset
2. Preprocesses the data by handling missing values, encoding categorical variables, and scaling numerical features
3. Applies SMOTE to balance the dataset
4. Trains three different machine learning models
5. Evaluates each model using multiple performance metrics
6. Generates visualization plots
7. Identifies the best-performing model

Output Files Generated

The script generates three visualization files:
roc_curves.png – ROC curves for all models
confusion_matrices.png – Confusion matrices comparison
model_comparison.png – Bar chart comparing model performance

Models Used

Logistic Regression – A fast and reliable baseline model
Random Forest – An ensemble model that improves accuracy and reduces overfitting
XGBoost – A powerful gradient boosting algorithm that typically provides strong performance on structured data

Future Improvements

In the future, I plan to add more machine learning models and apply hyperparameter tuning techniques to further improve performance and compare results more extensively.

Requirements

Python 3.8+
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
