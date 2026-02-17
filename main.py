"""
Employee Attrition Analysis
A comprehensive data analysis project exploring factors influencing employee attrition
using Logistic Regression and Decision Tree models.

Dataset: IBM HR Analytics Employee Attrition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

# Load the dataset
# Note: Update the file path to your CSV location
employee = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# View basic structure
print("Dataset Shape:", employee.shape)
print("\nDataset Info:")
print(employee.info())
print("\nFirst few rows:")
print(employee.head())
print("\nBasic Statistics:")
print(employee.describe())

# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES CHECK")
print("="*50)
print("Missing values per column:")
print(employee.isnull().sum())
print(f"\nTotal missing values: {employee.isnull().sum().sum()}")

# Check for duplicate rows
print("\n" + "="*50)
print("DUPLICATE CHECK")
print("="*50)
print(f"Number of duplicate rows: {employee.duplicated().sum()}")
print(f"Number of duplicate EmployeeNumbers: {employee['EmployeeNumber'].duplicated().sum()}")

# Check unique values for categorical variables
print("\n" + "="*50)
print("CATEGORICAL VARIABLES - UNIQUE VALUES")
print("="*50)
categorical_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Over18']
for col in categorical_cols:
    print(f"\n{col}: {employee[col].unique()}")

# Check ranges for numerical variables
print("\n" + "="*50)
print("NUMERICAL VARIABLES - RANGES")
print("="*50)
numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
                 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
                 'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                 'YearsWithCurrManager', 'TrainingTimesLastYear']

for col in numerical_cols:
    print(f"{col}: Min={employee[col].min()}, Max={employee[col].max()}")

# Check ranges for ordinal variables
print("\n" + "="*50)
print("ORDINAL VARIABLES - RANGES")
print("="*50)
ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 
               'JobLevel', 'JobSatisfaction', 'PerformanceRating', 
               'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

for col in ordinal_cols:
    print(f"{col}: Min={employee[col].min()}, Max={employee[col].max()}")

# =============================================================================
# DATA CLEANING
# =============================================================================

print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Identify constant columns
print("\nChecking for constant columns:")
print(f"EmployeeCount unique values: {employee['EmployeeCount'].unique()}")
print(f"StandardHours unique values: {employee['StandardHours'].unique()}")
print(f"Over18 unique values: {employee['Over18'].unique()}")

# Check correlation between rate variables and MonthlyIncome
print("\nCorrelation with MonthlyIncome:")
print(f"DailyRate: {employee['DailyRate'].corr(employee['MonthlyIncome']):.4f}")
print(f"HourlyRate: {employee['HourlyRate'].corr(employee['MonthlyIncome']):.4f}")
print(f"MonthlyRate: {employee['MonthlyRate'].corr(employee['MonthlyIncome']):.4f}")

# Remove unnecessary columns
columns_to_remove = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 
                    'Over18', 'DailyRate', 'HourlyRate', 'MonthlyRate']
employee = employee.drop(columns=columns_to_remove)

print(f"\nColumns after removal: {employee.shape[1]}")
print(f"Remaining columns: {list(employee.columns)}")

# Convert to appropriate datatypes
# Convert Attrition to categorical with specific order
employee['Attrition'] = pd.Categorical(employee['Attrition'], 
                                      categories=['No', 'Yes'], 
                                      ordered=True)

# Convert other categorical variables
categorical_features = ['BusinessTravel', 'Department', 'EducationField', 
                       'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
for col in categorical_features:
    employee[col] = employee[col].astype('category')

print("\nData types after conversion:")
print(employee.dtypes)

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)

# Separate numeric and categorical columns
numeric_cols = employee.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = employee.select_dtypes(include=['category', 'object']).columns.tolist()

# Numeric variables summary
print("\nNumeric Variables Summary:")
print(employee[numeric_cols].describe())

# Key variables detailed summary
key_vars = ['Age', 'MonthlyIncome', 'YearsAtCompany', 
           'TotalWorkingYears', 'JobSatisfaction', 'WorkLifeBalance']
print("\nKey Variables Summary:")
print(employee[key_vars].describe())

# Frequency tables for categorical variables
print("\n" + "="*50)
print("CATEGORICAL VARIABLES FREQUENCY")
print("="*50)
important_cats = ['Attrition', 'Department', 'JobRole', 'Gender', 'OverTime']
for col in important_cats:
    print(f"\n{col}:")
    print(employee[col].value_counts())

# =============================================================================
# DATA VISUALIZATION
# =============================================================================

print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Histograms for numeric variables
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Key Numeric Variables', fontsize=16, fontweight='bold')

axes[0, 0].hist(employee['Age'], bins=20, color='lightblue', edgecolor='white')
axes[0, 0].set_title('Histogram of Age')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(employee['MonthlyIncome'], bins=20, color='lightgreen', edgecolor='white')
axes[0, 1].set_title('Histogram of Monthly Income')
axes[0, 1].set_xlabel('Monthly Income')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(employee['YearsAtCompany'], bins=20, color='orange', edgecolor='white')
axes[1, 0].set_title('Histogram of Years at Company')
axes[1, 0].set_xlabel('Years at Company')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(employee['TotalWorkingYears'], bins=20, color='pink', edgecolor='white')
axes[1, 1].set_title('Histogram of Total Working Years')
axes[1, 1].set_xlabel('Total Working Years')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('histograms_numeric_variables.png', dpi=300, bbox_inches='tight')
plt.show()

# Bar plots for categorical variables
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Distribution of Categorical Variables', fontsize=16, fontweight='bold')

# Attrition count
attrition_counts = employee['Attrition'].value_counts()
axes[0].bar(attrition_counts.index, attrition_counts.values, 
           color=['lightgray', 'tomato'])
axes[0].set_title('Employee Attrition Count')
axes[0].set_xlabel('Attrition (Yes/No)')
axes[0].set_ylabel('Number of Employees')

# Department count
dept_counts = employee['Department'].value_counts()
axes[1].bar(range(len(dept_counts)), dept_counts.values, color='skyblue')
axes[1].set_title('Employees by Department')
axes[1].set_xlabel('Department')
axes[1].set_ylabel('Count')
axes[1].set_xticks(range(len(dept_counts)))
axes[1].set_xticklabels(dept_counts.index, rotation=45, ha='right')

# OverTime count
overtime_counts = employee['OverTime'].value_counts()
axes[2].bar(overtime_counts.index, overtime_counts.values, 
           color=['lightgreen', 'salmon'])
axes[2].set_title('Overtime Status')
axes[2].set_xlabel('OverTime')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('barplots_categorical_variables.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplots: Numeric vs Categorical
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Numeric Variables by Attrition Status', fontsize=16, fontweight='bold')

# Monthly Income by Attrition
employee.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[0])
axes[0].set_title('Monthly Income by Attrition')
axes[0].set_xlabel('Attrition (Yes/No)')
axes[0].set_ylabel('Monthly Income')
axes[0].get_figure().suptitle('')

# Age by Attrition
employee.boxplot(column='Age', by='Attrition', ax=axes[1])
axes[1].set_title('Age by Attrition')
axes[1].set_xlabel('Attrition')
axes[1].set_ylabel('Age')
axes[1].get_figure().suptitle('')

# Job Satisfaction by Attrition
employee.boxplot(column='JobSatisfaction', by='Attrition', ax=axes[2])
axes[2].set_title('Job Satisfaction by Attrition')
axes[2].set_xlabel('Attrition')
axes[2].set_ylabel('Job Satisfaction (1-4)')
axes[2].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('boxplots_by_attrition.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatterplots for relationships
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Relationships between Variables', fontsize=16, fontweight='bold')

axes[0].scatter(employee['TotalWorkingYears'], employee['MonthlyIncome'], 
               alpha=0.5, color='steelblue')
axes[0].set_title('Income vs Total Working Years')
axes[0].set_xlabel('Total Working Years')
axes[0].set_ylabel('Monthly Income')

axes[1].scatter(employee['YearsAtCompany'], employee['MonthlyIncome'], 
               alpha=0.5, color='coral')
axes[1].set_title('Years at Company vs Income')
axes[1].set_xlabel('Years at Company')
axes[1].set_ylabel('Monthly Income')

plt.tight_layout()
plt.savefig('scatterplots_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# HANDLING OUTLIERS
# =============================================================================

print("\n" + "="*50)
print("OUTLIER DETECTION AND REMOVAL")
print("="*50)

def get_outlier_indices(series):
    """Identify outliers using IQR method"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = (series < lower_bound) | (series > upper_bound)
    return series[outliers].index.tolist()

# Check outliers for key numeric variables
out_age = get_outlier_indices(employee['Age'])
out_income = get_outlier_indices(employee['MonthlyIncome'])
out_years_company = get_outlier_indices(employee['YearsAtCompany'])

print(f"Number of outliers in Age: {len(out_age)}")
print(f"Number of outliers in MonthlyIncome: {len(out_income)}")
print(f"Number of outliers in YearsAtCompany: {len(out_years_company)}")

# Visualize outliers with boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Boxplots for Outlier Detection', fontsize=16, fontweight='bold')

axes[0].boxplot(employee['Age'], vert=False)
axes[0].set_title('Boxplot of Age')
axes[0].set_xlabel('Age')

axes[1].boxplot(employee['MonthlyIncome'], vert=False)
axes[1].set_title('Boxplot of Monthly Income')
axes[1].set_xlabel('Monthly Income')

axes[2].boxplot(employee['YearsAtCompany'], vert=False)
axes[2].set_title('Boxplot of Years at Company')
axes[2].set_xlabel('Years at Company')

plt.tight_layout()
plt.savefig('boxplots_outlier_detection.png', dpi=300, bbox_inches='tight')
plt.show()

# Create cleaned dataset (removing outliers from MonthlyIncome)
employee_data_clean = employee.drop(index=out_income).reset_index(drop=True)

print(f"\nOriginal dataset shape: {employee.shape}")
print(f"Cleaned dataset shape: {employee_data_clean.shape}")

# Verify boxplot after removing outliers
plt.figure(figsize=(10, 4))
plt.boxplot(employee_data_clean['MonthlyIncome'], vert=False)
plt.title('Monthly Income (After Removing Outliers)', fontsize=14, fontweight='bold')
plt.xlabel('Monthly Income')
plt.savefig('boxplot_income_after_cleaning.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# DATA AGGREGATION
# =============================================================================

print("\n" + "="*50)
print("DATA AGGREGATION")
print("="*50)

# Mean Monthly Income by Attrition
agg_income_attrition = employee_data_clean.groupby('Attrition')['MonthlyIncome'].mean()
print("\nMean Monthly Income by Attrition:")
print(agg_income_attrition)

# Mean Job Satisfaction by Attrition
agg_js_attrition = employee_data_clean.groupby('Attrition')['JobSatisfaction'].mean()
print("\nMean Job Satisfaction by Attrition:")
print(agg_js_attrition)

# Mean Work-Life Balance by Attrition
agg_wlb_attrition = employee_data_clean.groupby('Attrition')['WorkLifeBalance'].mean()
print("\nMean Work-Life Balance by Attrition:")
print(agg_wlb_attrition)

# Mean Monthly Income by Department
agg_income_dept = employee_data_clean.groupby('Department')['MonthlyIncome'].mean()
print("\nMean Monthly Income by Department:")
print(agg_income_dept)

# Attrition rate by Department
attrition_by_dept = pd.crosstab(employee_data_clean['Department'], 
                                employee_data_clean['Attrition'])
print("\nAttrition Count by Department:")
print(attrition_by_dept)

# Advanced aggregation using groupby
print("\n" + "="*50)
print("COMPREHENSIVE AGGREGATION BY ATTRITION")
print("="*50)
agg_by_attrition = employee_data_clean.groupby('Attrition').agg({
    'Age': 'mean',
    'MonthlyIncome': 'mean',
    'JobSatisfaction': 'mean',
    'WorkLifeBalance': 'mean',
    'YearsAtCompany': 'mean',
    'EmployeeNumber': 'count'  # Using any column to count
}).round(2)
agg_by_attrition.columns = ['avg_age', 'avg_income', 'avg_job_satisfaction', 
                            'avg_work_life_balance', 'avg_years_at_company', 'n_employees']
print(agg_by_attrition)

# Aggregation by Department and Attrition
print("\n" + "="*50)
print("AGGREGATION BY DEPARTMENT AND ATTRITION")
print("="*50)
agg_dept_attrition = employee_data_clean.groupby(['Department', 'Attrition']).agg({
    'MonthlyIncome': 'mean',
    'JobSatisfaction': 'mean',
    'Age': 'count'
}).round(2)
agg_dept_attrition.columns = ['avg_income', 'avg_job_satisfaction', 'count']
print(agg_dept_attrition)

# =============================================================================
# MACHINE LEARNING - PREPARE DATA
# =============================================================================

print("\n" + "="*50)
print("MACHINE LEARNING PREPARATION")
print("="*50)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Create a copy for modeling
employee_ml = employee.copy()

# Convert Attrition to binary numeric
employee_ml['Attrition_num'] = (employee_ml['Attrition'] == 'Yes').astype(int)

# Encode categorical variables
label_encoders = {}
categorical_features = ['BusinessTravel', 'Department', 'EducationField', 
                       'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

for col in categorical_features:
    le = LabelEncoder()
    employee_ml[col + '_encoded'] = le.fit_transform(employee_ml[col])
    label_encoders[col] = le

# Train-Test Split
X_cols = ['JobSatisfaction', 'MonthlyIncome', 'WorkLifeBalance', 
         'JobInvolvement', 'Age', 'DistanceFromHome']
y = employee_ml['Attrition_num']
X = employee_ml[X_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=123, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Attrition rate in training: {y_train.mean():.2%}")
print(f"Attrition rate in test: {y_test.mean():.2%}")

# =============================================================================
# QUESTION 1: WHAT FACTORS MOST INFLUENCE EMPLOYEE ATTRITION?
# LOGISTIC REGRESSION MODEL 1
# =============================================================================

print("\n" + "="*50)
print("QUESTION 1: FACTORS INFLUENCING ATTRITION")
print("LOGISTIC REGRESSION MODEL")
print("="*50)

# Build logistic regression model
log_model1 = LogisticRegression(random_state=123, max_iter=1000)
log_model1.fit(X_train, y_train)

# Model coefficients
coef_df = pd.DataFrame({
    'Feature': X_cols,
    'Coefficient': log_model1.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nModel Coefficients:")
print(coef_df)

# Predictions
y_pred_proba1 = log_model1.predict_proba(X_test)[:, 1]
y_pred1 = (y_pred_proba1 > 0.5).astype(int)

# Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1)
print("\nConfusion Matrix:")
print(cm1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred1))

# ROC Curve
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_proba1)
auc1 = auc(fpr1, tpr1)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'ROC curve (AUC = {auc1:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Attrition Logistic Regression', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curve_logistic1.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAUC Score: {auc1:.4f}")

# =============================================================================
# QUESTION 2: HOW DOES WORK EXPERIENCE OR DEPARTMENT RELATE TO ATTRITION?
# LOGISTIC REGRESSION MODEL 2
# =============================================================================

print("\n" + "="*50)
print("QUESTION 2: DEPARTMENT AND EXPERIENCE ANALYSIS")
print("LOGISTIC REGRESSION MODEL")
print("="*50)

# Prepare data with department encoding
X_cols2 = ['Department_encoded', 'YearsAtCompany', 'TotalWorkingYears']
X2 = employee_ml[X_cols2]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.3, 
                                                        random_state=123, stratify=y)

# Build logistic regression model
log_model2 = LogisticRegression(random_state=123, max_iter=1000)
log_model2.fit(X_train2, y_train2)

# Model coefficients
coef_df2 = pd.DataFrame({
    'Feature': X_cols2,
    'Coefficient': log_model2.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nModel Coefficients:")
print(coef_df2)

# Predictions
y_pred_proba2 = log_model2.predict_proba(X_test2)[:, 1]
y_pred2 = (y_pred_proba2 > 0.5).astype(int)

# Confusion Matrix
cm2 = confusion_matrix(y_test2, y_pred2)
print("\nConfusion Matrix:")
print(cm2)

print("\nClassification Report:")
print(classification_report(y_test2, y_pred2))

# ROC Curve
fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_pred_proba2)
auc2 = auc(fpr2, tpr2)

plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'ROC curve (AUC = {auc2:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Department Analysis', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curve_logistic2.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAUC Score: {auc2:.4f}")

# =============================================================================
# DECISION TREE MODEL FOR ATTRITION
# =============================================================================

print("\n" + "="*50)
print("DECISION TREE MODEL")
print("="*50)

# Prepare comprehensive feature set
dt_features = ['Age', 'Gender_encoded', 'MaritalStatus_encoded', 'DistanceFromHome',
              'Education', 'Department_encoded', 'JobRole_encoded', 'JobLevel',
              'MonthlyIncome', 'OverTime_encoded', 'BusinessTravel_encoded',
              'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany',
              'TotalWorkingYears', 'PerformanceRating']

X_dt = employee_ml[dt_features]
y_dt = employee_ml['Attrition_num']

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.3, random_state=123, stratify=y_dt
)

# Build decision tree
dt_model = DecisionTreeClassifier(random_state=123, max_depth=5, 
                                 min_samples_split=20, min_samples_leaf=10,
                                 ccp_alpha=0.02)
dt_model.fit(X_train_dt, y_train_dt)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': dt_features,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.title('Top 10 Feature Importances - Decision Tree', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# Predictions
y_pred_dt = dt_model.predict(X_test_dt)
y_pred_proba_dt = dt_model.predict_proba(X_test_dt)[:, 1]

# Confusion Matrix
cm_dt = confusion_matrix(y_test_dt, y_pred_dt)
print("\nConfusion Matrix:")
print(cm_dt)

print("\nClassification Report:")
print(classification_report(y_test_dt, y_pred_dt))

# ROC Curve
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test_dt, y_pred_proba_dt)
auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='red', lw=2, label=f'ROC curve (AUC = {auc_dt:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Decision Tree Model', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curve_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAUC Score: {auc_dt:.4f}")

# Visualize Decision Tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
         feature_names=dt_features,
         class_names=['No', 'Yes'],
         filled=True,
         rounded=True,
         fontsize=10)
plt.title('Decision Tree for Employee Attrition', fontsize=16, fontweight='bold')
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression 1', 'Logistic Regression 2', 'Decision Tree'],
    'AUC': [auc1, auc2, auc_dt]
})

print(model_comparison)

# Visualize model comparison
plt.figure(figsize=(10, 6))
plt.bar(model_comparison['Model'], model_comparison['AUC'], 
       color=['blue', 'green', 'red'], alpha=0.7)
plt.ylim([0.5, 1.0])
plt.ylabel('AUC Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=15, ha='right')
for i, v in enumerate(model_comparison['AUC']):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nAll visualizations have been saved as PNG files.")
print("Check your working directory for the output files.")