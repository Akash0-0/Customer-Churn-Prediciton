# Customer-Churn-Prediciton
Bank Customer Churn Prediction using Machine Learning
# Bank Customer Churn Prediction

This project builds a Decision Tree classifier to predict customer churn for a bank.

## Features Used
- Customer demographics (age, gender, etc.)
- Account information (balance, tenure, number of products)
- Transaction behavior
- Customer service interaction
- Credit and loan status

## Model
- Decision Tree Classifier
- Hyperparameter tuning (optional)
- Handles class imbalance using class weights

## Performance Metrics
- Accuracy: ~81%
- F1-score for churn class: ~0.64
- Precision and recall analyzed with confusion matrix and classification report

## How to Use

1. Prepare your dataset as `X` (features) and `y` (target churn labels).
2. Split data into train and validation sets.
3. Run the training script `train_model.py`.
4. The trained model will be saved as `Customer-Churn-Predictor.pkl`.

## Requirements
- Python 3.x
- scikit-learn
- pandas, numpy (for data processing)

## Example Usage

```bash
python train_model.py


## CODE


---

### Python script: `train_model.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load your dataset here
# Replace 'your_data.csv' with your actual data file
data = pd.read_csv('your_data.csv')

# Assume last column is target 'Churn', adjust as necessary
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Initialize Decision Tree with class_weight to handle imbalance
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Print evaluation
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Save the trained model
joblib.dump(model, 'Customer-Churn-Predictor.pkl')
print("Model saved as Customer-Churn-Predictor.pkl")


AUTHOR
Aakash pal
palaakaah148@gmail.com
