#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataPreprocessing import X_train, X_test, y_train,y_test


#Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_GNB = classifier.predict(X_test)

#Confusion Matrix for Gaussian Naive Bayes 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_GNB)

#Accuracy for Gaussian Naive Bayes 
from sklearn.metrics import accuracy_score
ac_score_GNB = accuracy_score(y_test,y_pred_GNB)

#Precision for Gaussian Naive Bayes 
from sklearn.metrics import precision_score
p_score_GNB = precision_score(y_test,y_pred_GNB)

#Recall for Gaussian Naive Bayes 
from sklearn.metrics import recall_score
r_score_GNB = recall_score(y_test,y_pred_GNB)

#F1-Score for Gaussian Naive Bayes 
from sklearn.metrics import f1_score
f1_score_GNB = f1_score(y_test,y_pred_GNB)

#ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score_GNB = roc_auc_score(y_test,y_pred_GNB)

#Log Loss
from sklearn.metrics import log_loss
log_loss_GNB = log_loss(y_test,y_pred_GNB)

