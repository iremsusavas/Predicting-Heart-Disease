#Random Forest
#Support Vector Classifier

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataPreprocessing import X_train, X_test, y_train,y_test


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=50) #for 50 trees
#classifier = RandomForestClassifier(n_estimators=100) #for 100 trees
classifier = RandomForestClassifier(n_estimators=10) #for 10 trees
classifier.fit(X_train,y_train)
y_pred_RF = classifier.predict(X_test)

#Confusion Matrix for Random Forest
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_RF)

#Accuracy for Random Forest
from sklearn.metrics import accuracy_score
ac_score_RF = accuracy_score(y_test,y_pred_RF)

#Precision for Random Forest
from sklearn.metrics import precision_score
p_score_RF = precision_score(y_test,y_pred_RF)

#Recall for Random Forest
from sklearn.metrics import recall_score
r_score_RF = recall_score(y_test,y_pred_RF)

#F1-Score for Random Forest
from sklearn.metrics import f1_score
f1_score_RF = f1_score(y_test,y_pred_RF)

#ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score_RF = roc_auc_score(y_test,y_pred_RF)

#Log Loss
from sklearn.metrics import log_loss
log_loss_RF = log_loss(y_test,y_pred_RF)
