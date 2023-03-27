## Loan classification: Logistic regression, Random Forest, XGBoost, and SVM.

This project use machine learning methods to classify whether to give the loan for each observation. It aims to help the banking industry to be more efficient
in saving the time and the cost for reviewing large amount of applications.

## Data

### EDA
![loan_status](https://user-images.githubusercontent.com/119982930/226212934-e552e75f-20e1-4208-88e0-9c5a37501ec4.png)
![cat_var](https://user-images.githubusercontent.com/119982930/226212928-0ed7d1f7-0ad4-4041-abb9-3d52f6a275e7.png)
![con_var](https://user-images.githubusercontent.com/119982930/226212932-42cfa172-0fc8-4319-932c-d0694c8eb73a.png)


### Data preprocessing
1.) drop unnecessary column.

2.) missing values from both continuous and categorical variables are replaced by mode since it is an appropraite treatment for categorical variable or when the variable is skewed. 

### feature selection
This step is included to choose the best features which are correlated to Loan_status. 
It is important to differentiate between categorical variables ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'Loan_Amount_Term', 'Credit_History', and 'Property_Area') and continuous variables ('ApplicantIncome', 'CoapplicantIncome', and 'LoanAmount') since
there different methods to visulize and measure between these two type of variables. 

There are two methods that have been done in the script.

1.) Visualization

Bar plots: Categorical Vs Categorical(Loan_Status).

![cat_loan](https://user-images.githubusercontent.com/119982930/226212926-54ef88b2-4a79-4c74-8d29-f7bc8c9e4117.png)


Box plots: Continuous VS Categorical (Loan_Status).

![con_loan](https://user-images.githubusercontent.com/119982930/226212931-cffe5293-cfb5-41aa-9adc-768de4ba94e6.png)


2.) Statistical Feature Selection

However, to give more precise correlation, statistical tests are used to confirm the visualization.
**Chi-Square test** is used to check the correlation between two categorical varaibles. However, **ANOVA test** is used to check the correlation between continuous variable and categorical variable. The assumption for both test is based on the following

H0: the two variables are not correlated 

H1: the two varaibles are correlated

If p-value is <0.05, the H0 is rejected and two variables are correlated. 

The result from Chi-square test shows that 'Married', 'Education', 'Credit_History', and 'Property_Area' are correlated to Loan_Status. On the other hand, the result from ANOVA test shows that 'ApplicantIncome', 'CoapplicantIncome', and 'LoanAmount' are not correlated to Loan_Status.

In conclusion, only 'Married', 'Education', 'Credit_History', and 'Property_Area' variables are correlated to Loan_Status.

## Visualization

### The feature importance for Top 10 most important columns from Random Forest

![featureimpr_rf](https://user-images.githubusercontent.com/119982930/226212942-6361f9c1-35f3-49fa-b101-f4de37e4a763.png)

### Shap summary plot 

![shap_xgb](https://user-images.githubusercontent.com/119982930/226212935-ef8e277e-278a-49fa-b890-f53475b0661d.png)

Features in the summary plot are ranked by the means of SHAP values or how much impact each feature has on heart disease on average. Each dot represents each observation in the test set, and the color of each dot shows the effect of the feature value on heart disease when the feature value is increasing and decreasing. The dots that have been piled up vertically show the density of the observations in that sub-feature. As a result, the SHAP summary plot shows not only the range and the distribution of each feature on loan status on the global scale, but also shows the relationship between the feature value on loan status.

The two plots show that credit history is the most important feature to loan status. Howover, SHAP summary plot indicates that if observation is married and living in semiurban area, they will have an increase in a log odds ratio or more chance to get approved for the loan on average.

## Result

All algorithms have produced similar kind of average accuracy which is  0.74.

## Conclusion

Although all four models give high accuracy, a larger data set is needed to train the model in order to improve accuracy. 
