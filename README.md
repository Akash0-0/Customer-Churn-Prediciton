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

AUTHOR
Aakash pal
palaakaah148@gmail.com
