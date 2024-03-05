# Risk Analysis for Loan Default Prediction
This project aims to leverage Exploratory Data Analysis (EDA)and machine learning to conduct risk analysis for loan default prediction in the context of a consumer finance company. By analyzing historical loan application data, we will identify patterns and factors that indicate whether a client is likely to default on their loan payments. This analysis will assist the company in minimizing financial losses while ensuring that creditworthy applicants are not unfairly rejected.

## Data Exploration and Preprocessing

In this phase, we perform an initial exploration of the dataset and preprocess the data to prepare it for model training. The following steps are performed:

1. Importing libraries and setting up the notebook: We import the necessary libraries for data analysis and set up the Jupyter notebook.

2. Reading data and knowing the basics: We load the dataset and examine its shape and basic information using the Pandas library.

3. Checking for duplicated rows and null values: We check if there are any duplicated rows or null values in the dataset and handle them accordingly.

4. Exploring unique values: We explore the unique values in each column of the dataset to gain insights into the data distribution and identify any abnormalities.

5. Client Profile Analysis:: Analysis the client profile based on their income,education and employment

6. Financial Analysis: Analysis the Financial based on the income and dedt

7. Credit History Analysis:Analysis the relationships between the client's credit risk and the reasons for rejection of their previous loan application

8. Loan Application Decisions Analysis:Analysis the Loan Application Decisions based on credit_amount,different types of loans (cash vs. revolving),previous application outcomes (approval, refusal).

## Model Building and Evaluation
1. Train-Validation-Test Split:
   - Splitting the dataset into train, validation, and test sets

2. Scaling the Data:
   - Standard scaling of numerical columns to ensure consistency

3. Handling Class Imbalance:
   - Checking the class distribution in the training set
   - Addressing class imbalance using Synthetic Minority Over-sampling Technique (SMOTE)

4. Model :
   Given  the dataset,use SVM Classifier for Risk Analysis for Loan Default Prediction

5. Hyperparameter Tuning:
   - Using GridSearchCV to find the best hyperparameters for the SVM Classifier

## Model Performance on Test Set
1. Evaluation Metrics:
   - Calculating accuracy, precision, recall, and F1-score on the test set
   - Focusing on positive outcomes for evaluation

2. Results Interpretation:
   - Analyzing the performance metrics to assess the model's ability to Risk Analysis for Loan Default Prediction

