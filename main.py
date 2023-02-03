from data_prep import DataPreP
from sklearn.linear_model import LogisticRegression
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

val_pred = log_reg.predict(X_val)

val_acc = accuracy_score(y_val, val_pred)
print("Validation Accuracy: ", np.round(val_acc, 3))

val_prec = precision_score(y_val, val_pred)
print("Validation Precision: ", np.round(val_prec, 3))

val_rec = recall_score(y_val, val_pred)
print("Validation Recall: ", np.round(val_rec, 3))

val_auc_roc = roc_auc_score(y_val, val_pred)
print("ROC AUC Recall: ", np.round(val_auc_roc, 3))


#   Regularized Logistic Regression Model
print("Regularized Logistic Regression")
log_reg_l2 = LogisticRegression(penalty="l2")

log_reg_l2.fit(X_train, y_train)

val_pred = log_reg_l2.predict(X_val)

val_acc = accuracy_score(y_val, val_pred)
print("Validation Accuracy: ", np.round(val_acc, 3))

val_prec = precision_score(y_val, val_pred)
print("Validation Precision: ", np.round(val_prec, 3))

val_rec = recall_score(y_val, val_pred)
print("Validation Recall: ", np.round(val_rec, 3))

val_auc_roc = roc_auc_score(y_val, val_pred)
print("ROC AUC Recall: ", np.round(val_auc_roc, 3))