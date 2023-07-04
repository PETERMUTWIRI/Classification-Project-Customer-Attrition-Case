
# Classification-Project-Customer-Turnover-Case

power bi dashboard by peter mutwiri

https://app.powerbi.com/view?r=eyJrIjoiYjkzMjc0NDUtNzEwMi00NTgzLWE3MGItYzQ1OTg3NjEwNjcwIiwidCI6IjQ0ODdiNTJmLWYxMTgtNDgzMC1iNDlkLTNjMjk4Y2I3MTA3NSJ9
Inroduction

In today’s highly competitive business landscape, customer churn (also known as customer turnover or attrition) poses a significant challenge for companies across various industries. Customer retention has become a crucial aspect of business success, as losing valuable customers can have a detrimental impact on revenue and growth. To mitigate the risk of customer churn, organizations need to proactively identify and understand the factors that contribute to customer attrition.

This classification project focuses on analyzing customer churn and developing effective strategies for customer retention. By leveraging supervised machine learning techniques, we aim to identify key indicators and patterns that precede customer churn. Through the analysis of diverse data variables, we will build models that can predict churn with high accuracy, enabling businesses to take proactive measures to retain valuable customers.Get ready to dive into an immersive journey of knowledge and discovery as we delve into the fascinating world of customer churn with python code snippets explained with their respective comments.

In this Project, you will:

    Learn more about classification models and help the client, a telecommunication company, to understand their data.
    Find the lifetime value of each customer.
    know what factors affect the rate at which customers stop using their network.
    predict if a customer will churn or not

Skill sets you will build:

1. Data Exploration
2. Missing value computations
3. Feature engineering
4. Model Development using Machine Learning Algorithms like Logistic Regression, 5 5. Decision Trees, Support Vector Machine, Random Forest etc.
6. Model Evaluation and Interpretation using LIME, SHAP techniques
7 . Model Optimization and hyperparameter tuning
Catalogue:

*Installation and importation of important features

*Data Loading

* Data Evaluation (Eda)

* Data processing and Engineering

* Hypothesis Test

* Answering Questions with Visualizations

*Power Bi Deployment

* Balancing Dataset

* Train and Evaluate Four Models

* Evaluate Chosen Model

* Advance Model improvement

* Future Predictions

*Key Insight and Conclusions

Installation and importation of useful libraries :

import sqlalchemy as sa
import pyodbc
from dotenv import dotenv_values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import os
import pickle
import shutil

import warnings
warnings.filterwarnings("ignore")

Data Loading:

In order to kickstart our data analysis journey, we need to first load the data from multiple sources into our project environment. This section focuses on the data loading process, where we will retrieve data from three distinct sources: a database, GitHub, and OneDrive.

Loading Data from a Database: The first source of our data lies within a database. We have two options for establishing a connection and retrieving the required data: using python module we imported earlier pyodbc or sqlalchemy. With pyodbc, we can connect to the database using the appropriate credentials and establish a connection wich is stores in a .env file at the root of a repository by aid of dotenv module. Alternatively, sqlalchemy provides an abstraction layer for connecting to the database and simplifies the process of querying data.

# Load environment variables from .env file into a dictionary
environment_variables = dotenv_values('.env')


# Get the values for the credentials you set in the '.env' file
database = environment_variables.get("DATABASE")
server = environment_variables.get("SERVER")
username = environment_variables.get("USERNAME")
password = environment_variables.get("PASSWORD")
# Establishing the connection
connection_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

# Establish the connection using SQLAlchemy
engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")
# Create a connection
connection = engine.connect()

Once connected, we will write a query with table name parameters to extract the necessary data. This query will allow us to retrieve specific columns, filter data based on certain conditions, and aggregate information as required for our analysis. By leveraging the power of SQL queries, we can efficiently extract the relevant data from the database.Fetched data is stored in a DataFrame for accesibility.

# Execute the query
query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
result = connection.execute(query)
# Fetch all the rows from the result set into a Pandas DataFrame
df = pd.DataFrame(result.fetchall(), columns=result.keys())

Loading Data from GitHub: The second source of our data is GitHub, a popular platform for version control and collaborative development. GitHub allows users to host and share datasets, making it a valuable resource for accessing publicly available data. We utilize GitHub’s functionality to download the dataset or directly access the raw data via its URL.After downloading ,pandas module will perform its magic by reading the csv file upon provision of the file path.

# reading csv data
df1 = pd.read_csv('C:\\Users\\X1 CARBON\\Downloads\\LP2_Telco-churn-last-2000.csv')

Loading Data from OneDrive: The third source of our data is OneDrive, a cloud-based storage service provided by Microsoft. OneDrive offers a convenient platform for storing and sharing files, including datasets.Similar to GitHub, OneDrive allows us to either download the dataset directly or access it via a shareable link.

#reading a xlsx file from the local storage
df_test = pd.read_excel('C:\\Users\\X1 CARBON\\Downloads\\Telco-churn-second-2000.xlsx')

By leveraging data from these diverse sources, including databases, GitHub, and OneDrive, we ensure a comprehensive and varied dataset with a rich and extensive foundation to explore, analyze, and derive meaningful insights from our project

Hypothesis and questions statement:

HYPOTHESIS !

NULL HYPOTHESIS : H0

ALTERNATE HYPOTHESIS : H1

H0 — There is no significant difference in churn rates between male and female customers. H1 -Is there a significant difference in churn rates between male and female customers?

H0 : The type of internet service does not influence customer churn.

H1 : Does the type of internet service influence customer churn?

H0 :Customers with a longer tenure more likely to churn.

H1 : Are customers with a longer tenure less likely to churn?

H0 — The payment method does not influence customer churn.

H1- Does the payment method influence customer churn?

QUESTIONS ?

1. Is there a significant difference in churn rates between male and female customers?

2. Does the type of internet service influence customer churn?

3. Are customers with a longer tenure less likely to churn?

4. Does the payment method influence customer churn?
Data evaluation (EDA):

Exploratory Data Analysis (EDA) serves as the crucial first step in understanding the dataset and uncovering valuable insights. In this section, we will conduct a comprehensive evaluation of the data to gain a deeper understanding of the variables, their distributions, relationships, and potential patterns related to customer churn.

Overview of the Dataset: We will start by providing an overview of the dataset, including its size, structure, and the variables it contains.

#show information about the dataset
df_train.info()

output

<class 'pandas.core.frame.DataFrame'>
Int64Index: 5043 entries, 0 to 2042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        5043 non-null   object 
 1   gender            5043 non-null   object 
 2   SeniorCitizen     5043 non-null   object 
 3   Partner           5043 non-null   object 
 4   Dependents        5043 non-null   object 
 5   tenure            5043 non-null   int64  
 6   PhoneService      5043 non-null   object 
 7   MultipleLines     4774 non-null   object 
 8   InternetService   5043 non-null   object 
 9   OnlineSecurity    4392 non-null   object 
 10  OnlineBackup      4392 non-null   object 
 11  DeviceProtection  4392 non-null   object 
 12  TechSupport       4392 non-null   object 
 13  StreamingTV       4392 non-null   object 
 14  StreamingMovies   4392 non-null   object 
 15  Contract          5043 non-null   object 
 16  PaperlessBilling  5043 non-null   object

From the provided output, we can derive the following information:

    Dataset Size: The DataFrame consists of 5043 entries (rows) with 21 columns.
    Column Names: The DataFrame contains several columns, including ‘customerID’, ‘gender’, ‘SeniorCitizen’, ‘Partner’, ‘Dependents’, ‘tenure’, ‘PhoneService’, ‘MultipleLines’, ‘InternetService’, ‘OnlineSecurity’, ‘OnlineBackup’, ‘DeviceProtection’, ‘TechSupport’, ‘StreamingTV’, ‘StreamingMovies’, ‘Contract’, and ‘PaperlessBilling’.
    Data Types: The data types of the columns vary, including objects (strings) and int64 (integer).
    Non-Null Counts: The ‘customerID’, ‘gender’, ‘SeniorCitizen’, ‘Partner’, ‘Dependents’, ‘tenure’, ‘PhoneService’, ‘InternetService’, and ‘Contract’ columns have no missing values (non-null count equal to the total number of entries, 5043). However, other columns such as ‘MultipleLines’, ‘OnlineSecurity’, ‘OnlineBackup’, ‘DeviceProtection’, ‘TechSupport’, ‘StreamingTV’, ‘StreamingMovies’, and ‘PaperlessBilling’ have missing values (non-null count is less than 5043).
    Categorical Variables: The DataFrame includes categorical variables such as ‘gender’, ‘SeniorCitizen’, ‘Partner’, ‘Dependents’, ‘PhoneService’, ‘MultipleLines’, ‘InternetService’, ‘OnlineSecurity’, ‘OnlineBackup’, ‘DeviceProtection’, ‘TechSupport’, ‘StreamingTV’, ‘StreamingMovies’, ‘Contract’, and ‘PaperlessBilling’. These variables likely represent different categories or options for each customer.
    Numerical Variable: The ‘tenure’ column represents a numerical variable indicating the number of months the customer has been with the company.

By examining the DataFrame’s structure, column names, data types, non-null counts, and the presence of categorical and numerical variables, we gain an initial understanding of the dataset’s composition.

Description of the dataframe:Using .describe() methord to see the preview of distribution ,nature and unique values in columns of the dataframe

df_train.describe()

output




      customerID  gender  SeniorCitizen Partner Dependents  tenure  \
0     5600-PDUJF    Male              0      No         No       6   
1     8292-TYSPY    Male              0      No         No      19   
2     0567-XRHCU  Female              0     Yes        Yes      69   
3     1867-BDVFH    Male              0     Yes        Yes      11   
4     2067-QYTCF  Female              0     Yes         No      64   
...          ...     ...            ...     ...        ...     ...   
2038  6840-RESVB    Male              0     Yes        Yes      24   
2039  2234-XADUH  Female              0     Yes        Yes      72   
2040  4801-JZAZL  Female              0     Yes        Yes      11   
2041  8361-LTMKD    Male              1     Yes         No       4   
2042  3186-AJIEK    Male              0      No         No      66   

     PhoneService     MultipleLines InternetService OnlineSecurity  ...  \
0             Yes                No             DSL             No  ...   
1             Yes                No             DSL             No  ...   
2              No  No phone service             DSL            Yes  ...   
3             Yes               Yes     Fiber optic             No  ...   
4             Yes               Yes     Fiber optic             No  ...   
...           ...               ...             ...            ...  ...   
2038          Yes               Yes             DSL            Yes  ...   
2039          Yes               Yes     Fiber optic             No  ...   
2040           No  No phone service             DSL            Yes  ...   
2041          Yes               Yes     Fiber optic             No  ...   
2042          Yes                No     Fiber optic            Yes  ...   

     DeviceProtection TechSupport StreamingTV StreamingMovies        Contract  \
0                  No         Yes          No              No  Month-to-month   
1                 Yes         Yes          No              No  Month-to-month   
2                 Yes          No          No             Yes        Two year   
3                  No          No          No              No  Month-to-month   
4                 Yes         Yes         Yes             Yes  Month-to-month   
...               ...         ...         ...             ...             ...   
2038              Yes         Yes         Yes             Yes        One year   
2039              Yes          No         Yes             Yes        One year   
2040               No          No          No              No  Month-to-month   
2041               No          No          No              No  Month-to-month   
2042              Yes         Yes         Yes             Yes        Two year   

     PaperlessBilling              PaymentMethod MonthlyCharges  TotalCharges  \
0                 Yes    Credit card (automatic)          49.50         312.7   
1                 Yes    Credit card (automatic)          55.00        1046.5   
2                 Yes    Credit card (automatic)          43.95        2960.1   
3                 Yes           Electronic check          74.35         834.2   
4                 Yes           Electronic check         111.15        6953.4   
...               ...                        ...            ...           ...   
2038              Yes               Mailed check          84.80        1990.5   
2039              Yes    Credit card (automatic)         103.20        7362.9   
2040              Yes           Electronic check          29.60        346.45   
2041              Yes               Mailed check          74.40         306.6   
2042              Yes  Bank transfer (automatic)         105.65        6844.5   

     Churn  
0       No  
1      Yes  
2       No  
3      Yes  
4       No  
...    ...  
2038    No  
2039    No  
2040    No  
2041   Yes  
2042    No  

From the above otput we can derive that 75% customers have tenure less than 56 months and Average monthly charge $65.one should go farther analyzing using .describe methord on all the columns individually for deeply understanding the data.

Missing values evaluation.Handling missing values in a DataFrame is an essential step in data preprocessing to ensure the accuracy and reliability of subsequent analyses. later on the project we will handle missing values using several techniques.Below is a code to output missing values.

df_train.isna().sum()

output

customerID            0
gender                0
SeniorCitizen         0
Partner               0
Dependents            0
tenure                0
PhoneService          0
MultipleLines       269
InternetService       0
OnlineSecurity      651
OnlineBackup        651
DeviceProtection    651
TechSupport         651
StreamingTV         651
StreamingMovies     651
Contract              0
PaperlessBilling      0
PaymentMethod         0
MonthlyCharges        0
TotalCharges          8
Churn                 1
dtype: int64

we can see a total of 8 columns have missing values

Univariate analysis of numerical columns: is a statistical analysis technique that focuses on examining and understanding individual variables in isolation. It provides insights into the distribution, central tendency, dispersion, and other important characteristics of a single variable.we will plot visualizations of individual variable by use of seabon and matplotlib.These are modules used in making visualivisualisations.

# Univariate Analysis
# Numeric Columns
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_train[numeric_columns].hist(bins=30, figsize=(15, 10))
plt.suptitle('Univariate Analysis - Numeric Columns', y=0.92)
plt.show()

Based on the histograms above we drow the following conclusions.Tenure Distribution:

The histogram for the ‘tenure’ column shows that the distribution is slightly right-skewed, with a peak around the lower values. This indicates that a significant number of customers have a relatively shorter tenure with the company.

Monthly Charges Distribution: The histogram for the ‘MonthlyCharges’ column suggests a bimodal distribution. There are two peaks, indicating that there may be distinct groups of customers with different monthly charges. Further analysis is needed to explore the factors contributing to this pattern.

Total Charges Distribution: The histogram for the ‘TotalCharges’ column shows a positively skewed distribution, with a long tail towards higher values. This suggests that there is a group of customers with higher total charges, potentially indicating long-term or high-value customers.

univariate analysis of categorical columns:

# Categorical Columns(Univariate)
categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
# ploting visualisations
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df_train)
    plt.title(f'Univariate Analysis - {column}')
    plt.xticks(rotation=45)
    plt.show()

This is just a sample plot of the categorical univariate analysis showing the distribution of male and female on thegender column by count.The code above should provide plots for all categorical univariate analysis .

Bivariate analysis:Bivariate analysis is a statistical method used to analyze the relationship between two variables. It examines how changes in one variable affect the other variable. In the code below, bivariate analysis is performed on categorical variables in the dataset with respect to the “Churn” variable.

# Bivariate Analysis
for column in categorical_columns:
    if column != 'Churn':
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, hue='Churn', data=df_train)
        plt.title(f'Bivariate Analysis: {column} vs. Churn')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Churn', labels=['No', 'Yes'])
        plt.show()

Below is one sample output of the code

In the context of customer churn, the countplot would show the distribution of churned and non-churned customers across different genders.we get insight that female has slightly higher amount of churn than male indicating a potential relationship between gender and customer retention.

Multivariate analysis:Multivariate analysis is a statistical technique used to analyze the relationships between multiple variables simultaneously. It explores how changes in multiple variables are associated with each other and how they collectively impact dependent variable of interest in our case is the churn variable.

# Multivariate Analysis
# Numeric Columns
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
plt.figure(figsize=(10, 8))
sns.heatmap(df_train[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Multivariate Analysis - Numeric Columns')
plt.show()

we can derive that the heatmap’s diagonal pattern with correlation coefficients of 1 suggests that there is a strong positive linear relationship between ‘tenure’ and ‘MonthlyCharges’, and between ‘MonthlyCharges’ and ‘TotalCharges’. It implies that customers with longer tenure tend to have higher monthly charges and higher total charges, while customers with shorter tenure have lower monthly charges and lower total charges

HYPOTHESIS TEST

A hypothesis test is a statistical procedure used to make inferences or draw conclusions about a population based on sample data. It involves formulating a null hypothesis (H0) and an alternative hypothesis (Ha) and using statistical methods to determine whether the evidence from the data supports rejecting the null hypothesis in favor of the alternative hypothesis.

H0 — There is no significant difference in churn rates between male and female customers.

H1 -Is there a significant difference in churn rates between male and female customers?

Chi-Square Test of Independence:is a specific type of hypothesis test used to assess the independence between two categorical variables. The null hypothesis states that there is no association or relationship between the variables, while the alternative hypothesis suggests that there is a significant association.

contingency_table = pd.crosstab(df_train['gender'], df_train['Churn'])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Set the significance level
alpha = 0.05

# Print the results
print("Chi-Square Test of Independence - Gender vs. Churn")
print(f"Chi2 statistic: {chi2}")
print(f"P-value: {p_value}")

# Check the result based on the significance level
if p_value < alpha:
    print("There is a significant association between gender and churn.")
else:
    print("There is no significant association between gender and churn.")

output

Chi-Square Test of Independence - Gender vs. Churn
Chi2 statistic: 0.021628520637713346
P-value: 0.8830796247912641
There is no significant association between gender and churn.

The Chi2 statistic is a measure of the discrepancy between the observed frequencies and the frequencies that would be expected if the two variables (gender and churn) were independent. A smaller Chi2 statistic indicates a smaller discrepancy.

Our p-value (0.8) is greater than the commonly used significance level of 0.05, we fail to reject the null hypothesis. This means that there is not enough evidence to suggest a significant association between gender and churn. Therefore, we can conclude that gender does not have a significant influence on customer churn based on the available data.

H0 : The type of internet service does not influence customer churn.

H1 : Does the type of internet service influence customer churn?

contingency_table = pd.crosstab(df_train['InternetService'], df_train['Churn'])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Set the significance level
alpha = 0.05

# Print the results
print("Chi-Square Test of Independence - Internet Service vs. Churn")
print(f"Chi2 statistic: {chi2:.2f}")
print(f"P-value: {p_value:.2f}")

# Check the result based on the significance level
if p_value < alpha:
    print("There is a significant association between Internet Service and Churn.")
else:
    print("There is no significant association between Internet Service and Churn.")

output

Chi-Square Test of Independence - Internet Service vs. Churn
Chi2 statistic: 562.27
P-value: 0.00
There is a significant association between Internet Service and Churn

The chi2 statistic measures the degree of association between the variables (Internet Service and Churn)

the p-value is extremely small (0.00), significantly lower than the common significance level of 0.05, we reject the null hypothesis. This means that there is strong evidence to suggest a significant association between Internet Service and Churn. The result indicates that the type of Internet Service a customer has is likely to have an impact on their likelihood to churn.

H0 :Customers with a longer tenure more likely to churn.

H1 : Are customers with a longer tenure less likely to churn?

Hypothesis Test: T-test-Ttest is a statistical test used to determine if there is a significant difference between the means of two groups or populations.

group1 = df_train[df_train['Churn'] == 'No']['tenure'].values
group2 = df_train[df_train['Churn'] == 'Yes']['tenure'].values

t_statistic, p_value = ttest_ind(group1, group2)

# Set the significance level
alpha = 0.05

# Print the results
print("Independent t-test - Tenure vs. Churn")
print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.2f}")

# Check the result based on the significance level
if p_value < alpha:
    print("There is a significant difference in tenure between churned and non-churned customers.")
else:
    print("There is no significant difference in tenure between churned and non-churned customers.")

output

Independent t-test - Tenure vs. Churn
T-statistic: 26.59
P-value: 0.00
There is a significant difference in tenure between churned and non-churned customers.

The t-statistic value is 26.60. This value represents the magnitude of the difference in the average tenure between churned and non-churned customers.

The p-value is 0.00. This value represents the probability of obtaining the observed difference in tenure between churned and non-churned customers by chance alone. In this case, the p-value is extremely small, indicating strong evidence against the null hypothesis (no difference).

H0 — The payment method does not influence customer churn.

H1- Does the payment method influence customer churn?

contingency_table = pd.crosstab(df_train['PaymentMethod'], df_train['Churn'])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Set the significance level
alpha = 0.05

# Print the results
print("Chi-Square Test of Independence - Payment Method vs. Churn")
print(f"Chi2 statistic: {chi2:.2f}")
print(f"P-value: {p_value:.2f}")

# Check the result based on the significance level
if p_value < alpha:
    print("There is a significant association between Payment Method and Churn.")
else:
    print("There is no significant association between Payment Method and Churn.")

output

Chi-Square Test of Independence - Payment Method vs. Churn
Chi2 statistic: 435.18
P-value: 0.00
There is a significant association between Payment Method and Churn.

A Chi2 statistic of 434.58 indicates a stronger association

The p-value of (0.00) represents the probability of observing the association between Payment Method and Churn by chance alone.

we can conclude that there is a significant association between Payment Method and Churn. The choice of Payment Method appears to have an influence on the likelihood of a customer churning.
Answering Questions with Visualizations

Question 1: Is there a significant difference in churn rates between male and female customers?

# Create a contingency table of gender and churn
contingency_table = pd.crosstab(df_train['gender'], df_train['Churn'])

# Calculate the churn rates by gender
churn_rates = contingency_table['Yes'] / contingency_table.sum(axis=1)

# Plot the churn rates
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_rates.index, y=churn_rates.values)
plt.title('Churn Rates by Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
plt.show()

plot

The female churn rate is approximately 0.25 and the male churn rate is approximately 0.23, it suggests that there is a slight difference in churn rates between the two genders. However, the difference is relatively small.

This means that, on average, the churn rate for female customers is slightly higher than the churn rate for male customers, but the difference is not substantial. It indicates that gender alone may not be a strong predictor of customer churn in this dataset.

Question 2 : Does the type of internet service influence customer churn

plt.figure(figsize=(8, 6))
sns.countplot(x='InternetService', hue='Churn', data=df_train)
plt.title('Churn by Internet Service')
plt.xlabel('Internet Service')
plt.ylabel('Count')
plt.show()

The results of the visualization indicate that customers with DSL internet service have the highest number of non-churned customers, with a count of 1400. On the other hand, the count of churned customers for DSL internet service is 300.

This suggests that customers with DSL internet service are less likely to churn compared to other internet service types. Optics, on the other hand, has a lower count of non-churned customers and a higher count of churned customers.

In summary, the type of internet service does seem to influence customer churn, with DSL internet service showing a lower churn rate compared to other types.

Question 3 — Are customers with a longer tenure less likely to churn

# Calculate churn rate for different tenure groups
tenure_groups = df_train.groupby(pd.cut(df_train['tenure'], bins=[0, 12, 24, 36, 48, 60, float('inf')]))
churn_rate = tenure_groups['Churn'].value_counts(normalize=True).unstack()['Yes']

# Plot the churn rate
plt.figure(figsize=(10, 6))
churn_rate.plot(kind='bar')
plt.title('Churn Rate by Tenure')
plt.xlabel('Tenure Groups')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45)
plt.show()

Customers with a tenure of 0–12 months have the highest churn rate of 0.4. This indicates that customers who are relatively new to the service are more likely to churn. It could be because they are still in the early stages of evaluating the service or may have encountered issues that led to dissatisfaction.

Customers with a tenure of 24–36 months have a lower churn rate of 0.2. This suggests that customers who have been with the service for a moderate amount of time are less likely to churn compared to those with shorter tenures.

Customers with a tenure of 60 months and above have the lowest churn rate of 0.05. This indicates that customers who have been with the service for a longer duration are significantly less likely to churn. These customers have likely developed a strong relationship with the service provider, are satisfied with the service, and may have a higher level of loyalty.

In summary, the results suggest that there is a correlation between tenure and churn rate. Customers with longer tenures are indeed less likely to churn compared to those with shorter tenures.

Question 4 : Does the payment method influence customer churn?

plt.figure(figsize=(10, 8))
sns.countplot(x='PaymentMethod', hue='Churn', data=df_train)
plt.title('Churn by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.show()

The results of the visualization indicate that the payment method of “Electronic check” has the highest count of both non-churned and churned customers, with approximately 900 non-churned and 750 churned customers. On the other hand, the payment methods “Mail check,” “Bank transfer (automatic),” and “Credit card (automatic)” have a relatively similar number of customers at around 900, but their churn count is significantly lower at approximately 200.

This suggests that the payment method of “Electronic check” has a higher churn rate compared to other payment methods. Customers who use electronic checks as their payment method are more likely to churn compared to customers who use other payment methods such as mail check, bank transfer (automatic), or credit card (automatic).

In summary, the payment method does seem to influence customer churn, with electronic check users having a higher churn rate compared to other payment methods.
Feature Processing & Engineering

Feature processing and engineering are essential steps in the data preprocessing phase of machine learning and data analysis. These steps involve transforming and creating new features from the raw data to improve the performance and effectiveness of the models.

Feature processing refers to the process of preparing the raw data by applying various techniques such as cleaning, scaling, normalization, and handling missing values. It aims to ensure that the data is in a suitable format for further analysis and modeling

Dropping missing value on churn column since its only one

df_train.dropna(subset=['Churn'], inplace=True)

Converting the Target variable to a binary numeric value Yes:1, No:0 since machine learning models work with numeric variables

# Create a NumPy array to represent the target variable
target = np.where(df_train['Churn'] == 'Yes', 1, 0)

# Assign the new NumPy array to the 'Churn' column
df_train['Churn'] = target

# Verify the conversion
df_train['Churn'].unique 

Drop the CustomerID column because it relevance is limited

df_train.drop("customerID", axis=1, inplace=True)

Replacing missing values in specific columns

The rationale behind the code is to replace missing values in certain columns with “No” based on the condition that the corresponding customer does not have an internet service. This is because, without internet service, the specific services (e.g., online security, device protection) are not applicable or available.

columns_to_replace = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for column in columns_to_replace:
    df_train[column] = np.where((df_train["InternetService"] == "No") & (df_train[column].isnull()), "No internet service", df_train[column])

Defining categorical and numerical values to be used on the next phase

categorical = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines', 
               'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
               'StreamingTV','StreamingMovies', 'Contract','PaperlessBilling','PaymentMethod']
numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']

Dataset Splitting

Dataset splitting is an important step in machine learning to properly evaluate and validate the performance of a trained model. The process involves dividing the available dataset into multiple subsets: typically a training set, a validation set, and a test set.

X = df_train.drop(columns="Churn") creates a new DataFrame X by dropping the column "Churn" from the original df_train. This will include all the columns except for the "Churn" column, representing the features or independent variables.

y = df_train["Churn"].values creates a NumPy array y containing the values of the "Churn" column from the original df_train. This represents the target variable or the dependent variable that we want to predict.

#creating dataframe x and y
X = df_train.drop(columns = "Churn")
y = df_train["Churn"].values

#splittin the y and x datasets to both train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4, stratify =y)

On the above code we use train_test_split function from the scikit-learn library to split the features (X) and target variable (y) into training and testing sets.

    X: The features or independent variables.
    y: The target variable or dependent variable.
    test_size: The proportion of the data to be allocated for the testing set. In this case, it is set to 0.2, which means that 20% of the data will be used for testing, and the remaining 80% will be used for training.
    random_state: The seed value used by the random number generator for reproducibility. Setting it to a specific value (e.g., 4) ensures that the same random split is obtained every time you run the code.
    stratify: This parameter is set to y, which ensures that the splitting is stratified based on the values of y. It helps to maintain the proportion of different classes in the target variable across the training and testing sets. This is particularly useful when dealing with imbalanced datasets where the classes are not equally represented

Impute Missing Values

This is the phase of now filling the missing values, a process known as imputation

impute missing values with the mode for categorical features using scikit-learn’s SimpleImputer class.In this line, you create an instance of the SimpleImputer class and specify the imputation strategy as 'most_frequent'. This strategy indicates that the missing values should be replaced with the mode (most frequent value) of the column

Here, we apply the imputation to the training set. X_train[categorical] represents the subset of the training set that contains only the categorical features. The fit_transform method of the SimpleImputer is used to fit the imputer on the training data (X_train[categorical]) and simultaneously perform the imputation. It replaces the missing values in the categorical features with the mode and returns the imputed dataset

#Impute missing values with the mode for categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = categorical_imputer.fit_transform(X_train[categorical])
X_test_imputed = categorical_imputer.transform(X_test[categorical])
#Impute missing values with the mode for categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = categorical_imputer.fit_transform(X_train[categorical])
X_test_imputed = categorical_imputer.transform(X_test[categorical])

Feature Encoding

Feature encoding, also known as feature transformation or feature representation, is the process of converting categorical or textual data into numerical representations that machine learning algorithms can understand and process effectively. This is necessary because most machine learning algorithms are designed to work with numerical data.

There are several common techniques for feature encoding, depending on the nature of the data and the specific requirements of the problem:

One-Hot Encoding:One-hot encoding is used for categorical variables with no inherent ordinal relationship between categories. Each category is represented as a binary feature, where a value of 1 indicates the presence of the category, and 0 indicates its absence. This technique creates new binary columns for each category, effectively expanding the feature space.

Label Encoding:Label encoding is used when there is an ordinal relationship among the categories. Each category is assigned a unique integer label. For example, in an ordinal variable like “Low,” “Medium,” and “High,” the categories could be encoded as 0, 1, and 2, respectively.

Ordinal Encoding:Ordinal encoding is similar to label encoding but with a more explicit mapping of categories to ordinal values. It assigns numerical values to each category based on a predefined order or mapping. This can be useful when the categories have a natural order or hierarchy.other known techniques are Target Encoding,Binary Encoding and Feature Hashing

# Apply one-hot encoding on the training set
encoder = OneHotEncoder(drop='if_binary')
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical]).toarray(),
                               columns=encoder.get_feature_names_out(categorical),
                               index=X_train.index)

# Apply the same encoding on the testing set
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical]).toarray(),
                              columns=encoder.get_feature_names_out(categorical),
                              index=X_test.index)

Feature Scaling

Feature scaling, also known as feature normalization, is the process of transforming numerical features in a dataset to a common scale. It is performed to ensure that all features contribute equally to the analysis and modeling process, regardless of their original scales or units

Min-Max Scaling:Min-Max scaling, also known as normalization, scales the features to a specific range, typically between 0 and 1. It subtracts the minimum value of the feature and divides it by the range (maximum value minus the minimum value). The resulting transformed feature will have values between 0 and 1. Min-Max scaling preserves the original distribution of the feature .The other technique is Standardization (Z-score normalization)

# Create an instance of the StandardScaler
scaler = StandardScaler()

# Fit and transform the numeric variables in the training set
X_train_encoded[numerical] = scaler.fit_transform(X_train[numerical])

# Transform the numeric variables in the testing set using the fitted scaler
X_test_encoded[numerical] = scaler.transform(X_test[numerical])
# Concatenate the encoded & scaled training and testing sets
X_encoded = pd.concat([X_train_encoded, X_test_encoded], axis=0)
# Reset the index of X_train
X_train.reset_index(drop=True, inplace=True)

# Create a new DataFrame with the target variable and the "MonthlyCharges_TotalCharges_Ratio" column
correlation_df = pd.concat([X_train['MonthlyCharges_TotalCharges_Ratio'], pd.Series(y_train, name='Churn')], axis=1)

# Calculate the correlation matrix
correlation_matrix = correlation_df.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap: MonthlyCharges_TotalCharges_Ratio vs Churn')
plt.show()

The above The heatmap allows us to visually identify the strength and direction of the relationship between the variables. The intensity of the color in the heatmap indicates the magnitude of the correlation. Darker colors ( dark red) represent stronger correlations, while lighter colors ( light red) indicate weaker correlations.
Train set Balancing

When training machine learning models, it is important to ensure that the training data is balanced, especially if there is an imbalance in the distribution of the target variable. Imbalanced data can lead to biased models that perform poorly on the minority class.

# Convert y_train and y_test to Pandas Series
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

# Count the number of samples in each class in the training set
train_class_counts = y_train_series.value_counts()

# Count the number of samples in each class in the testing set
test_class_counts = y_test_series.value_counts()

# Plot the class distribution
plt.figure(figsize=(8, 6))
plt.bar(train_class_counts.index, train_class_counts.values, color='blue', label='Train')
plt.bar(test_class_counts.index, test_class_counts.values, color='orange', label='Test')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.legend()
plt.show()

This plot provides an overview of the class distribution in the dataset and helps in understanding the balance or imbalance among different classes.

Distribution of churn and non-churn instances in both the training and testing sets.

train_churn_count = pd.Series(y_train).value_counts()
test_churn_count = pd.Series(y_test).value_counts()

print("Churn count in y_train:")
print(train_churn_count)

print("\nChurn count in y_test:")
print(test_churn_count) 

output

Churn count in y_train:
0    2964
1    1069
dtype: int64

Churn count in y_test:
0    742
1    267
dtype: int64

Training set: The majority class is “0” (non-churn) with 2,964 instances, while the minority class is “1” (churn) with 1,069 instances. This indicates that the training set is imbalanced, with significantly more non-churn instances compared to churn instances.

Testing set: Similarly, the majority class is “0” (non-churn) with 742 instances, while the minority class is “1” (churn) with 267 instances. The class imbalance is also observed in the testing set.

Now we applying the SMOTETomek algorithm to balance the training data. SMOTETomek is a combination of the SMOTE (Synthetic Minority Over-sampling Technique) and Tomek Links algorithms, which oversamples the minority class (churn) and undersamples the majority class (non-churn) simultaneously.

# Create an instance of SMOTETomek
smote_tomek = SMOTETomek()

# Apply SMOTETomek to the training data
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_encoded, y_train)

# Check the count of each class in the balanced training data
balanced_class_count = pd.Series(y_train_balanced).value_counts()
print("Class count in the balanced training data:")
print(balanced_class_countcode

The above code snippet creates an instance of the SMOTETomek class: smote_tomek = SMOTETomek().

The SMOTETomek algorithm is then applied to the training data using the fit_resample method: X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_encoded, y_train). This resamples the training data to create a balanced version with equal representation of the churn and non-churn classes.
Train Models / Evaluate / Model Comparison

Train the model: Use the training data to fit the model. This involves feeding the input data to the model and adjusting the model’s parameters based on the observed errors or loss. The optimization algorithm minimizes the loss function and updates the model’s parameters iteratively.

Evaluate the model: Assess the performance of the trained model using validation data or cross-validation. Calculate relevant metrics such as accuracy, precision, recall, F1-score, or mean squared error, depending on the problem type. Evaluate how well the model generalizes to unseen data and consider adjusting hyperparameters or model structure if necessary.

The code provided below defines a function called calculate_metrics that calculates performance metrics for a binary classification problem. It currently calculates the F1-score based on the true labels (y_true) and predicted labels (y_pred)

# Define the calculate_metrics function
def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['f1_score'] = f1_score(y_true, y_pred)
    return metrics

Then we initialise the models, we shall be using the following models Logistic Regression,Decision Tree,Random Forest,Gradient Boosting, AdaBoost, SVM,KNN, Naive Bayes, XGBoost, LightGBM, CatBoost the models will be build and evaluated with imbalance data ,the create a leaderboard that will show the metric perfomance of the models

`leaderboard_imbalanced = {}

for model, name in zip(models, model_names):
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)
    metrics = calculate_metrics(y_test, y_pred)
    leaderboard_imbalanced[name] = metrics
    
    # Print the classification report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['No', 'Yes']))
    print()

# Create a DataFrame from the leaderboard dictionary
leaderboard_df = pd.DataFrame(leaderboard_imbalanced).transpose()

# Format the F1-score column to display with two decimal places
leaderboard_df['Imbalanced Data F1-score'] = leaderboard_df['f1_score'].map('{:.2f}'.format)

# Print the leaderboard DataFrame
print("Leaderboard: Imbalance Data")
leaderboard_df.drop('f1_score', axis=1, inplace=True)
leaderboard_df

when we call the leaderboard dataframe we get an output showing the results of f1 scores of the models as follows .these results are of imbalanced data

 Imbalanced Data F1-score
Logistic Regression 0.60
Decision Tree 0.59
Random Forest 0.54
Gradient Boosting 0.60
AdaBoost 0.56
SVM 0.58
KNN 0.54
Naive Bayes 0.59
XGBoost 0.59
LightGBM 0.60
CatBoost 

As with imbalanced data we builds and evaluates the models on balanced training data. We then creates a leaderboard of model performance based on the F1-score metric, and finally prints the leaderboard DataFrame.

leaderboard_balanced = {}

for model, name in zip(models, model_names):
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_encoded)
    metrics = calculate_metrics(y_test, y_pred)
    leaderboard_balanced[name] = metrics
    
    # Print the classification report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['No', 'Yes']))
    print()

# Create a DataFrame from the leaderboard dictionary
leaderboard_df1 = pd.DataFrame(leaderboard_balanced).transpose()

 #Format the F1-score column to display with two decimal places
leaderboard_df1['Balanced Data F1-score'] = leaderboard_df1['f1_score'].map('{:.2f}'.format)

# Print the leaderboard DataFrame
print("Leaderboard: Balanced Data")
leaderboard_df1.drop('f1_score', axis=1, inplace=True)
leaderboard_df1

with balanced data we get the followinng results

 Balanced Data F1-score
Logistic Regression 0.63
Decision Tree 0.64
Random Forest 0.63
Gradient Boosting 0.64
AdaBoost 0.63
SVM 0.62
KNN 0.55
Naive Bayes 0.59
XGBoost 0.63
LightGBM 0.59
CatBoost 

We can see that Gradient Boosting has the highest fi score meaning it has higher correct prediction than other models
Confusion matrix

The confusion matrix, also known as the error matrix, is used to evaluate the performance of a machine learning model by examining the number of observations that are correctly and incorrectly classified. Each column of the matrix contains the predicted classes while each row represents the actual classes or vice versa. In a perfect classification, the confusion matrix will be all zeros except for the diagonal. All the elements out of the main diagonal represent misclassifications. It is important to bear in mind that the confusion matrix allows us to observe patterns of misclassification (which classes and to which extend they were incorrectly classified).

In binary classification problems, the confusion matrix is a 2-by-2 matrix composed of 4 elements:

    TP (True Positive): number of patients with spine problems that are correctly classified as sick.
    TN (True Negative): number of patients without pathologies who are correctly classified as healthy.
    FP (False Positive): number of healthy patients that are wrongly classified as sick.
    FN (False Negative): number of patients with spine diseases that are misclassified as healthy.

Now that the model is trained, it is time to evaluate its performance using the testing set. First, we use the previous model (gradient boosting classifier with best hyperparameters) to predict the class labels of the testing data (with the predict method). Then, we construct the confusion matrix using the confusion_matrix function from the sklearn.metrics package to check which observations were properly classified. The output is a NumPy array where the rows represent the true values and the columns the predicted classes.

# Iterate over each model and its corresponding name in the leaderboard
for model, name in zip(models, model_names):
    # Fit the model on the balanced training data
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predict the labels for the test data
    y_pred = model.predict(X_test_encoded)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted Churn")
    plt.ylabel("Actual Churn")
    plt.show()
    
    # Print the confusion matrix
    print(f"Confusion Matrix for {name}:")
    print(cm)
    print()

Hyperparameter tuning

Thus far we have split our data into a training set for learning the parameters of the model, and a testing set for evaluating its performance. The next step in the machine learning process is to perform hyperparameter tuning. The selection of hyperparameters consists of testing the performance of the model against different combinations of hyperparameters, selecting those that perform best according to a chosen metric and a validation method.

For hyperparameter tuning, we need to split our training data again into a set for training and a set for testing the hyperparameters (often called validation set). It is a very common practice to use k-fold cross-validation for hyperparameter tuning. The training set is divided again into k equal-sized samples, 1 sample is used for testing and the remaining k-1 samples are used for training the model, repeating the process k times. Then, the k evaluation metrics (in this case the accuracy) are averaged to produce a single estimator.

It is important to stress that the validation set is used for hyperparameter selection and not for evaluating the final performance of our model, as shown in the image below.

Hyperparameter tuning with cross-validation — Image created by the author

There are multiple techniques to find the best hyperparameters for a model. The most popular methods are (1) grid search, (2) random search, and (3) bayesian optimization. Grid search test all combinations of hyperparameters and select the best performing one. It is a really time-consuming method, particularly when the number of hyperparameters and values to try are really high.

In random search, you specify a grid of hyperparameters, and random combinations are selected where each combination of hyperparameters has an equal chance of being sampled. We do not analyze all combinations of hyperparameters, but only random samples of those combinations. This approach is much more computationally efficient than trying all combinations; however, it also has some disadvantages. The main drawback of random search is that not all areas of the grid are evenly covered, especially when the number of combinations selected from the grid is low.

Grid search vs random search — Image created by the author

We can implement random search in Scikit-learn using the RandomSearchCV class from the sklearn.model_selection package.

First of all, we specify the grid of hyperparameter values using a dictionary (grid_parameters) where the keys represent the hyperparameters and the values are the set of options we want to evaluate. Then, we define the RandomizedSearchCV object for trying different random combinations from this grid. The number of hyperparameter combinations that are sampled is defined in the n_iter parameter. Naturally, increasing n_iter will lead in most cases to more accurate results, since more combinations are sampled; however, on many occasions, the improvement in performance won’t be significant.

# Define the models and their respective parameter grids
models = {
    'AdaBoost': AdaBoostClassifier(random_state=4),
    'Logistic Regression': LogisticRegression(random_state=4),
    'Random Forest' : RandomForestClassifier(random_state=4),
    'Gradient Boosting' : GradientBoostingClassifier(random_state=4)
}

leaderboard = {}
for name, model in models.items():
    # Get the available parameters for the model
    available_params = model.get_params()
    print(f"Available parameters for {name}: {available_params}\n")

# Perform hyperparameter tuning and store the best-tuned models
leaderboard = {}

tuning using GridSearchCV and Setting the verbosity level for printing progress

for name, model in models.items():
    # Define the selected parameter values for the model
    
    param_selections = {
    'AdaBoost': {'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [0.1, 0.05, 0.01, 0.001]},
    'Logistic Regression': {'C': [0.1, 1, 10, 100, 1000], 'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']},
    'Random Forest':  {'n_estimators': [50, 100, 150, 200, 300, 400, 500], 'max_depth': [5, 10, 20]},
    'Gradient Boosting': {'n_estimators': [100, 150, 200, 250, 300], 'learning_rate': [0.1, 0.05, 0.01, 0.001], 'max_depth': [3, 5, 7]}
     }
    
     #Set the verbosity level for printing progress
    verbose = 3 if name == 'Logistic Regression' else 0
    
    # Get the selected parameter values for the model
    param_grid = param_selections[name]
    
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5, verbose=verbose, refit=True)
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters for {name}: {best_params}\n")
    
    
     # Predict using the best model
    y_pred = best_model.predict(X_test_encoded)
    f1 = f1_score(y_test, y_pred)
    f1_rounded = round(f1, 2)
    leaderboard[name] = {'Tuned F1-score': f1}

we created columns to store f1 cores of selected models with imbalanced data ,balanced data and after hyperparameter tuning now we will call them to see our progress

`# Filter leaderboard for selected models
selected_models = ['Gradient Boosting','Random Forest', 'Logistic Regression', 'AdaBoost']
filtered_df = leaderboard_df.loc[selected_models]

filtered_df1 = leaderboard_df1.loc[selected_models]

# Filter leaderboard_df2 for selected models
filtered_df2 = leaderboard_df2.loc[selected_models]

# Concatenate the filtered DataFrames
combined_df = pd.concat([filtered_df, filtered_df1, filtered_df2], axis=1)

# Print the combined DataFrame
combined_df

these our results

 Imbalanced Data F1-score Balanced Data F1-score Tuned F1-score
Gradient Boosting 0.60      0.64                     0.64
Random Forest 0.54          0.63                     0.61
Logistic Regression 0.60    0.63                     0.63
AdaBoost 0.56               0.63                     0.63

After tuning our models we can see that the best performing model is gradient booting with f1 score of 64%
Drawing conclusions — Summary

In this post, we have walked through a complete end-to-end machine learning project using the Telco customer Churn dataset. We started by cleaning the data and analyzing it with visualization. Then, to be able to build a machine learning model, we transformed the categorical data into numeric variables (feature engineering). After transforming the data, we tried 6 different machine learning algorithms using default parameters. Finally, we tuned the hyperparameters of the best performance model for model optimization, obtaining an accuracy of nearly 64%.

It is important to stress that the exact steps of a machine learning task vary by project. Although in the article we followed a linear process, machine learning projects tend to be iterative rather than linear processes, where previous steps are often revisited as we learn more about the problem we try to solve.
https://medium.com/@mutpeet/ml-classification-project-customer-turnover-case-5e7bd0a2d6c4
