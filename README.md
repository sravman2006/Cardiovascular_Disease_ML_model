# Cardiovascular disease prediction model. 
This project builds a machine learning model to assess the risk of cardiovascular disease using a large-scale health dataset with over 300,000 records. The model was trained on lifestyle, biometric, and medical history features to classify individuals as high or low risk for heart disease. 

Model Overview 
Algorithm: Random Forest Classifier
Training Data: Cleaned dataset with 300,000+ entries
Target Variable: Binary classification — Heart Disease (Yes/No)
Features: Includes age category, general health rating, exercise habits, diet, smoking history, diabetes, depression, BMI, and more

Performance Summary
The model achieves high accuracy (~92%), primarily due to strong performance on the majority class (low-risk individuals).
However, it shows bias toward predicting low risk, with limited sensitivity to high-risk cases.
Precision and recall for the positive class (heart disease) are low, indicating room for improvement in detecting true positives.

Limitations
Class imbalance in the dataset leads to underperformance on minority (high-risk) cases.
Recall for positive class is low, suggesting the model may miss many true cases of heart disease.
