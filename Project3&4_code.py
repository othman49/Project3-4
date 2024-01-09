#PROJECT 3 EDA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# Setting up the visualization
plt.style.use('seaborn-darkgrid')

# Creating subplots for key numerical variables
fig, ax = plt.subplots(2, 3, figsize=(18, 12))

# Age Distribution
sns.histplot(bank_data['age'], bins=30, kde=True, ax=ax[0, 0])
ax[0, 0].set_title('Age Distribution')

# Duration of the call Distribution
sns.histplot(bank_data['duration'], bins=30, kde=True, ax=ax[0, 1])
ax[0, 1].set_title('Call Duration Distribution')

# Campaign contacts Distribution
sns.histplot(bank_data['campaign'], bins=30, kde=True, ax=ax[0, 2])
ax[0, 2].set_title('Campaign Contacts Distribution')

# pdays Distribution
sns.histplot(bank_data[bank_data['pdays'] != 999]['pdays'], bins=30, kde=True, ax=ax[1, 0])  # 999 means client was not previously contacted
ax[1, 0].set_title('Days Since Last Contact Distribution')

# Previous contacts Distribution
sns.histplot(bank_data['previous'], bins=30, kde=True, ax=ax[1, 1])
ax[1, 1].set_title('Previous Contacts Distribution')

# Euribor 3 month rate Distribution
sns.histplot(bank_data['euribor3m'], bins=30, kde=True, ax=ax[1, 2])
ax[1, 2].set_title('Euribor 3 Month Rate Distribution')

plt.tight_layout()
plt.show()

# Calculating the counts for each job type
job_counts = data['job'].value_counts()

# Creating a pie chart for the 'job' column
plt.figure(figsize=(10, 10))
plt.pie(job_counts, labels=job_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
plt.title('Distribution of Job Types')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Preparing data for the 3D plot
unique_education = data['education'].unique()
education_mapping = {edu: i for i, edu in enumerate(unique_education)}
data['education_mapped'] = data['education'].map(education_mapping)

unique_marital = data['marital'].unique()
marital_mapping = {mar: i for i, mar in enumerate(unique_marital)}
data['marital_mapped'] = data['marital'].map(marital_mapping)

# Creating a 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['age'], data['education_mapped'], data['marital_mapped'], c=data['age'], cmap='viridis', marker='o')
ax.set_title('3D Visual of Age, Education, and Marital Status')
ax.set_xlabel('Age')
ax.set_ylabel('Education')
ax.set_zlabel('Marital Status')

# Customizing the ticks for education and marital status to show the actual names
ax.set_yticks(list(education_mapping.values()))
ax.set_yticklabels(list(education_mapping.keys()))
ax.set_zticks(list(marital_mapping.values()))
ax.set_zticklabels(list(marital_mapping.keys()))

# Adding a color bar to indicate age
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Age')
plt.show()

#PROJECT 4 PREDICTIVE MODELING 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initializing and training the Logistic Regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_pca, Y_train)

# Predicting on the test set
Y_pred_logistic = logistic_model.predict(X_test_pca)

# Evaluating the model
logistic_accuracy = accuracy_score(Y_test, Y_pred_logistic)
logistic_precision = precision_score(Y_test, Y_pred_logistic)
logistic_recall = recall_score(Y_test, Y_pred_logistic)
logistic_f1_score = f1_score(Y_test, Y_pred_logistic)
logistic_confusion_matrix = confusion_matrix(Y_test, Y_pred_logistic)

# Checking for overfitting: Comparing training and testing scores
logistic_train_accuracy = accuracy_score(Y_train, logistic_model.predict(X_train_pca))

logistic_metrics = {
    "Accuracy": logistic_accuracy,
    "Precision": logistic_precision,
    "Recall": logistic_recall,
    "F1 Score": logistic_f1_score,
    "Confusion Matrix": logistic_confusion_matrix.tolist(),
    "Training Accuracy": logistic_train_accuracy
from sklearn.naive_bayes import GaussianNB

# Initializing and training the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_pca, Y_train)

# Predicting on the test set
Y_pred_nb = nb_model.predict(X_test_pca)

# Evaluating the model
nb_accuracy = accuracy_score(Y_test, Y_pred_nb)
nb_precision = precision_score(Y_test, Y_pred_nb)
nb_recall = recall_score(Y_test, Y_pred_nb)
nb_f1_score = f1_score(Y_test, Y_pred_nb)
nb_confusion_matrix = confusion_matrix(Y_test, Y_pred_nb)

# Checking for overfitting: Comparing training and testing scores
nb_train_accuracy = accuracy_score(Y_train, nb_model.predict(X_train_pca))

nb_metrics = {
    "Accuracy": nb_accuracy,
    "Precision": nb_precision,
    "Recall": nb_recall,
    "F1 Score": nb_f1_score,
    "Confusion Matrix": nb_confusion_matrix.tolist(),
    "Training Accuracy": nb_train_accuracy
}

nb_metrics

from sklearn.tree import DecisionTreeClassifier

# Initializing and training the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_pca, Y_train)

# Predicting on the test set
Y_pred_dt = dt_model.predict(X_test_pca)

# Evaluating the model
dt_accuracy = accuracy_score(Y_test, Y_pred_dt)
dt_precision = precision_score(Y_test, Y_pred_dt)
dt_recall = recall_score(Y_test, Y_pred_dt)
dt_f1_score = f1_score(Y_test, Y_pred_dt)
dt_confusion_matrix = confusion_matrix(Y_test, Y_pred_dt)

# Checking for overfitting: Comparing training and testing scores
dt_train_accuracy = accuracy_score(Y_train, dt_model.predict(X_train_pca))

dt_metrics = {
    "Accuracy": dt_accuracy,
    "Precision": dt_precision,
    "Recall": dt_recall,
    "F1 Score": dt_f1_score,
    "Confusion Matrix": dt_confusion_matrix.tolist(),
    "Training Accuracy": dt_train_accuracy
}

dt_metrics
from sklearn.ensemble import RandomForestClassifier

# Initializing and training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, Y_train)

# Predicting on the test set
Y_pred_rf = rf_model.predict(X_test_pca)

# Evaluating the model
rf_accuracy = accuracy_score(Y_test, Y_pred_rf)
rf_precision = precision_score(Y_test, Y_pred_rf)
rf_recall = recall_score(Y_test, Y_pred_rf)
rf_f1_score = f1_score(Y_test, Y_pred_rf)
rf_confusion_matrix = confusion_matrix(Y_test, Y_pred_rf)

# Checking for overfitting: Comparing training and testing scores
rf_train_accuracy = accuracy_score(Y_train, rf_model.predict(X_train_pca))

rf_metrics = {
    "Accuracy": rf_accuracy,
    "Precision": rf_precision,
    "Recall": rf_recall,
    "F1 Score": rf_f1_score,
    "Confusion Matrix": rf_confusion_matrix.tolist(),
    "Training Accuracy": rf_train_accuracy
}
rf_metrics

from sklearn.ensemble import GradientBoostingClassifier

# Initializing and training the Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_pca, Y_train)

# Predicting on the test set
Y_pred_gb = gb_model.predict(X_test_pca)

# Evaluating the model
gb_accuracy = accuracy_score(Y_test, Y_pred_gb)
gb_precision = precision_score(Y_test, Y_pred_gb)
gb_recall = recall_score(Y_test, Y_pred_gb)
gb_f1_score = f1_score(Y_test, Y_pred_gb)
gb_confusion_matrix = confusion_matrix(Y_test, Y_pred_gb)

# Checking for overfitting: Comparing training and testing scores
gb_train_accuracy = accuracy_score(Y_train, gb_model.predict(X_train_pca))

gb_metrics = {
    "Accuracy": gb_accuracy,
    "Precision": gb_precision,
    "Recall": gb_recall,
    "F1 Score": gb_f1_score,
    "Confusion Matrix": gb_confusion_matrix.tolist(),
    "Training Accuracy": gb_train_accuracy
}
gb_metrics
from sklearn.svm import SVC

# Initializing and training the Support Vector Machine (SVM) model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_pca, Y_train)

# Predicting on the test set
Y_pred_svm = svm_model.predict(X_test_pca)

# Evaluating the model
svm_accuracy = accuracy_score(Y_test, Y_pred_svm)
svm_precision = precision_score(Y_test, Y_pred_svm)
svm_recall = recall_score(Y_test, Y_pred_svm)
svm_f1_score = f1_score(Y_test, Y_pred_svm)
svm_confusion_matrix = confusion_matrix(Y_test, Y_pred_svm)

# Checking for overfitting: Comparing training and testing scores
svm_train_accuracy = accuracy_score(Y_train, svm_model.predict(X_train_pca))

svm_metrics = {
    "Accuracy": svm_accuracy,
    "Precision": svm_precision,
    "Recall": svm_recall,
    "F1 Score": svm_f1_score,
    "Confusion Matrix": svm_confusion_matrix.tolist(),
    "Training Accuracy": svm_train_accuracy
}
svm_metrics
