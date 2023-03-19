
# Create a classification model which can predict whether to approve a loan application.

# library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load data set
df = pd.read_csv("/Users/ruochentan1/Desktop/loan.csv")
pd.set_option('display.expand_frame_repr', False) # expand all output
df.info() # check variable type and missing values( gender, married, dependents, self-employted.LoanAmount, Loan_Amount_Term)
df.head()
df.describe(include = "all").T # descriptive statistics of the data

# --------------  EDA
df = df.drop(columns=['Loan_ID']) # not useful to us
# 1.) bar plot of variable of interest (categorical variable)
sns.countplot(data=df, x="Loan_Status")
# 2.) histogram for categorical variable
categorical = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
fig,axes = plt.subplots(4,2,figsize=(15,13))
for idx,cat_col in enumerate(categorical):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=df,ax=axes[row,col])

plt.subplots_adjust(hspace=0.3)
fig.savefig("cat_var.png")
# 3.) histogram for continuous variable
numerical = ["CoapplicantIncome","LoanAmount","ApplicantIncome"]
fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical):
    sns.histplot(x=cat_col,data=df,ax=axes[idx],bins = 30) # change number of bins

print(df[numerical].describe())
fig.savefig("con_var.png")

# ----------- data preprocessing
# replace  mode  for continuous variable or LoanAmount since it is skewed and categorical variable.
df.isnull().sum()
df['LoanAmount'].fillna(df['LoanAmount'].mode()[0], inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# ----------- Feature selection: interpret by analyze which variable may be correlated with Loan_Status
# 1.) Loan_Status VS categorical variable
figr1,axes = plt.subplots(4,2,figsize=(15,13))
for idx,cat_col in enumerate(categorical):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=df,ax=axes[row,col],hue="Loan_Status")

plt.subplots_adjust(hspace=0.3)
figr1.savefig("cat_loan.png")

# 2.) Loan_Status VS continuous variable (when plot cat VS con use boxplot)
figr,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical):
    sns.boxplot(y=cat_col,data=df,x='Loan_Status',ax=axes[idx])

print(df[numerical].describe())
plt.subplots_adjust(wspace=0.3) # adjust the width between plot
figr.savefig("con_loan.png")


# confirm correlation between explanatory variables and target variable
# 1.) ANOVA is performed to check

def Anova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    predictors = []

    print('ANOVA Results\n')
    for predictor in ContinuousPredictorList:
        CategoryGroupLists = inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            predictors.append(predictor)
        else:
            print(predictor, 'is not correlated with', TargetVariable, '| P-Value:', AnovaResults[1])

    return (predictors)

# Calling the function to check which categorical variables are correlated with target
continuous=['ApplicantIncome', 'CoapplicantIncome','LoanAmount']
Anova(inpData=df, TargetVariable='Loan_Status', ContinuousPredictorList=continuous)


# 2.) Chi-Square test is conducted to check the correlation between two categorical variables

# Writing a function to find the correlation of all categorical variables with the Target variable
def Chisq(inpData, TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency

    predictors = []

    for predictor in CategoricalVariablesList:
        CrossTabResult = pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)

        if (ChiSqResult[1] < 0.05): # if p-value is <0.05, then we reject H0 or they are correlated
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            predictors.append(predictor)
        else:
            print(predictor, 'is not correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])

    return (predictors)

categorical=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area']

Chisq(inpData=df,
              TargetVariable='Loan_Status',
              CategoricalVariablesList= categorical)


# --------- selecting explanatory variable (only variables that are correlated to loan_status)
Selectedvariables=['Married', 'Education', 'Credit_History', 'Property_Area']

# Selecting final columns
df_final=df[Selectedvariables]
df_final2 = df_final.copy()
df_final.head()
df_final2.to_pickle('df_final.pkl')

# open the file by
import pickle
with open('df_final.pkl', 'rb') as f:
    df_ML = pickle.load(f)

#--------- data preprocessing for machine learing methods
# 1.)  convert binary nominal variables to numeric
df_ML['Married'].replace({'Yes':1, 'No':0}, inplace=True)
df_ML['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace=True)

#2.) Treating all the nominal variables at once using dummy variables
df_ML=pd.get_dummies(df_ML)

# Adding Target Variable to the data
df_ML['Loan_Status']=df['Loan_Status']
df_ML['Loan_Status'].replace({'Y':1, 'N':0}, inplace=True)

# check the result
df_ML.head()

# split test and train
TargetVariable=['Loan_Status'] # str
Predictors=['Married', 'Education', 'Credit_History', 'Property_Area_Rural',
       'Property_Area_Semiurban', 'Property_Area_Urban'] # list

X=df_ML[Predictors].values
y=df_ML[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

y_train= np.ravel(y_train)  # change format to row
y_test = np.ravel(y_test)

# Method ( logistic, random forest, XGBoost, SVM)
# --------------   (1) logistic regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(C=1,penalty='l2')
log=clf_log.fit(X_train,y_train)
prediction=log.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
# Overall Accuracy of the model
F1_Score= metrics.f1_score(y_test, prediction, average='weighted')
print("Overall accuracy on Testing Sample Data:", round(F1_Score,2))
print("Confusion Matrix on Test data")
pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True)

# 10-Fold Cross validation
from sklearn.model_selection import cross_val_score
Accuracy_Values=cross_val_score(log, X , np.ravel(y), cv=10, scoring='f1_weighted')
print("Average Accuracy of the model:", round(Accuracy_Values.mean(),3)) # average 0.784


# ---------------- (2) Random Forest (Bagging of multiple Decision Trees)
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth=3, n_estimators=50,criterion='gini',random_state= 123)

# Printing all the parameters of Random Forest
print(clf_rf)

# Creating the model on Training Data
rf=clf_rf.fit(X_train,y_train)
prediction=rf.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print("Overall accuracy on Testing Sample Data:", round(F1_Score,2))


# feature importance for random forest
feature_importances = pd.Series(rf.feature_importances_, index=Predictors)
feature_importances.nlargest(5).plot(kind='barh') # credit history is the most important variable for loan approval


# ------------------- (3) XGBoost
import xgboost
clf_xgb=xgboost.XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=50, objective='binary:logistic')

# Printing all the parameters of XGBoost
print(clf_xgb)

# Creating the model on Training Data
xgb=clf_xgb.fit(X_train,y_train)
prediction=xgb.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Plotting the feature importance for Top 10 most important columns
import shap
explainer = shap.TreeExplainer(clf_xgb)
shap_values = explainer.shap_values(df_ML.iloc[:,0:6])
shap.summary_plot(shap_values, df_ML.iloc[:,0:6])

# ----------------- (4) Support Vector Machines(SVM)
from sklearn import svm
clf_svm = svm.SVC(C=1, kernel='rbf', gamma=0.1)

# Printing all the parameters of KNN
print(clf_svm)

# Creating the model on Training Data
svm =clf_svm.fit(X_train,y_train)
prediction=svm.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print("Overall accuracy on Testing Sample Data:", round(F1_Score,2))

