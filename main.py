from data_prep import DataPreP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

dp = DataPreP("dv_logtistic")
X_train, y_train, X_val, y_val, X_test, y_test = dp.create_datasets("./pima-indians-diabetes.csv")

print({val:list(y_train).count(val) for val in [0, 1]})
print({val:list(y_val).count(val) for val in [0, 1]})
print({val:list(y_test).count(val) for val in [0, 1]})


#   Logistic Regression
print("Logistic Regression")
log_reg= LogisticRegression(penalty=None)
log_reg.fit(X_train, y_train)
print("\n*************************************")
print("Validation Data")
val_pred = log_reg.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("Validation Accuracy: ", np.round(val_acc, 3))
val_prec = precision_score(y_val, val_pred)
print("Validation Precision: ", np.round(val_prec, 3))
val_rec = recall_score(y_val, val_pred)
print("Validation Recall: ", np.round(val_rec, 3))
val_auc_roc = roc_auc_score(y_val, val_pred)
print("ROC AUC Score: ", np.round(val_auc_roc, 3))


test_pred = log_reg.predict(X_test)
print("\n*************************************")
print("Test Data")
test_acc = accuracy_score(y_test, test_pred)
print("Test Accuracy: ", np.round(test_acc, 3))
test_prec = precision_score(y_test, test_pred)
print("Test Precision: ", np.round(test_prec, 3))
test_rec = recall_score(y_test, test_pred)
print("Test Recall: ", np.round(test_rec, 3))
test_auc_roc = roc_auc_score(y_test, test_pred)
print("ROC AUC Score: ", np.round(test_auc_roc, 3))


#   Regularized Logistic Regression Model

print("\n*************************************")
print("Regularized Logistic Regression")

params = [{"penalty" :["l1", "l2", "elasticnet", None],
           "C": [1, 10, 100, 1000],
           "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
           "l1_ratio": np.linspace(0,1,11)}]

grid_search = GridSearchCV(
                estimator=LogisticRegression(),
                param_grid=params,
                scoring="accuracy",
                cv=10,
                verbose=1)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

log_reg_regularized = LogisticRegression(**grid_search.best_params_)
log_reg_regularized.fit(X_train, y_train)

val_pred = log_reg_regularized.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print("\n*************************************")
print("Validation Data")
print("Validation Accuracy: ", np.round(val_acc, 3))
val_prec = precision_score(y_val, val_pred)
print("Validation Precision: ", np.round(val_prec, 3))
val_rec = recall_score(y_val, val_pred)
print("Validation Recall: ", np.round(val_rec, 3))
val_auc_roc = roc_auc_score(y_val, val_pred)
print("ROC AUC Score: ", np.round(val_auc_roc, 3))

test_pred = log_reg_regularized.predict(X_test)
print("\n*************************************")
print("Test Data")
test_acc = accuracy_score(y_test, test_pred)
print("Test Accuracy: ", np.round(test_acc, 3))
test_prec = precision_score(y_test, test_pred)
print("Test Precision: ", np.round(test_prec, 3))
test_rec = recall_score(y_test, test_pred)
print("Test Recall: ", np.round(test_rec, 3))
test_auc_roc = roc_auc_score(y_test, test_pred)
print("ROC AUC Score: ", np.round(test_auc_roc, 3))


# Output
"""
{0: 347, 1: 190}
{0: 77, 1: 38}
{0: 76, 1: 40}
Logistic Regression

*************************************
Validation Data
Validation Accuracy:  0.774
Validation Precision:  0.773
Validation Recall:  0.447
ROC AUC Score:  0.691

*************************************
Test Data
Test Accuracy:  0.784
Test Precision:  0.759
Test Recall:  0.55
ROC AUC Score:  0.729

*************************************
Regularized Logistic Regression
Fitting 10 folds for each of 1056 candidates, totalling 10560 fits
{'C': 1, 'l1_ratio': 0.0, 'penalty': 'l2', 'solver': 'newton-cg'}

*************************************
Validation Data
Validation Accuracy:  0.765
Validation Precision:  0.762
Validation Recall:  0.421
ROC AUC Score:  0.678

*************************************
Test Data
Test Accuracy:  0.802
Test Precision:  0.793
Test Recall:  0.575
ROC AUC Score:  0.748

We can see that with the implemention of regularization (l2 regulatizer),
and k-fold (10) cross validation, we have checked the overfitting in the 
data and we can observe improvement in the accuracy, precision, recall and 
roc-auc score for the test data
"""