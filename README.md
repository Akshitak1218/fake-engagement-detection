# Fake Engagement Detection Using Machine Learning

## Overview

This project implements a machine learning–based approach to detect fake Instagram accounts using profile-level and behavioral features. Fake accounts contribute to misleading engagement metrics and reduce platform credibility. The objective of this project is to classify Instagram accounts as **fake** or **genuine** reliably and systematically.

## Dataset Description

The dataset consists of Instagram account metadata. Each record represents one account and is labeled as either genuine or fake.

### Input Features

* Profile picture availability
* Username and full name characteristics
* Biography length
* External URL presence
* Account privacy status
* Number of posts
* Number of followers
* Number of accounts followed

### Target Variable

* `fake`

  * 0 – Genuine account
  * 1 – Fake account

This is a binary classification problem.



## Methodology

The dataset was preprocessed by removing duplicate entries and handling missing values. An 80–20 train–validation split was applied to ensure unbiased evaluation. Feature scaling was performed where required.

Two classification models were implemented:

* **Logistic Regression** was used as a baseline model due to its simplicity and interpretability.
* **Random Forest Classifier** was used as the final model because of its ability to capture non-linear relationships and feature interactions.

Model performance was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. Feature importance analysis was conducted to identify the most influential predictors.



## Results

Logistic Regression achieved approximately 90% validation accuracy, serving as a strong baseline. The Random Forest model outperformed the baseline, achieving approximately 94% validation accuracy with balanced precision and recall. Behavioral features such as follower count and posting activity were found to be the most significant indicators of fake accounts.



## Execution Instructions

1. Install the required dependencies:

      pip install pandas numpy scikit-learn matplotlib seaborn
  

2. Run the script:

      python fake_engagement_detection.py
  

The script outputs validation results, displays a confusion matrix, and generates predictions for the test dataset in `test_predictions.csv`.



## Conclusion

This project demonstrates that machine learning techniques can effectively detect fake Instagram accounts using profile-based and behavioural features. The Random Forest model provided superior performance and reliable insights into engagement authenticity, making it suitable for practical fake engagement detection tasks.



