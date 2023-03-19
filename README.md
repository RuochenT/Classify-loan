## Loan classification: Logistic regression, Random Forest, XGBoost, and SVM.

This project use machine learning methods to classify whether to give the loan for each observation. It aims to help the banking industry to be more efficient
in saving the time and the cost for reviewing large amount of applications.

## Data

### EDA



### Data preprocessing
1.) drop unnecessary column (Loan_status).

2.) missing values treatment by replacing with mode of the column.

### feature selection
This step is included to choose the best features which are correlated to Loan_status. 
It is important to differentiate between categorical variables ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'Loan_Amount_Term', 'Credit_History', and 'Property_Area') and continuous variables ('ApplicantIncome', 'CoapplicantIncome', and 'LoanAmount') since
there different methods to visulize and measure between these two type of variables. 

There are two methods that have been done in the script.

1.) Visualization

Bar plots: Categorical Vs Categorical(Loan_Status).

Box plots: Continuous VS Categorical (Loan_Status).


2.) Statistical Feature Selection

However, to give more precise correlation, statistical tests are used to confirm the visualization.
**Chi-Square test** is used to check the correlation between two categorical varaibles. However, **ANOVA test** is used to check the correlation between continuous variable and categorical variable. The assumption for both test is based on the following

H0: the two variables are not correlated 

H1: the two varaibles are correlated

If p-value is <0.05, the H0 is rejected and two variables are correlated. 

The result from Chi-square test shows that 'Married', 'Education', 'Credit_History', and 'Property_Area' are correlated to Loan_Status. On the other hand, the result from ANOVA test shows that 'ApplicantIncome', 'CoapplicantIncome', and 'LoanAmount' are not correlated to Loan_Status.

In conclusion, only 'Married', 'Education', 'Credit_History', and 'Property_Area' variables are correlated to Loan_Status.

## Method and process 

## Result

The accuracy from four models are 0.74

## Conclusion

Although all four models give high accuracy, a larger data set is needed to train the model in order to improve accuracy. 
