#K-Nearest Neighbours

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataPreprocessing import X_train, X_test, y_train,y_test


#K Nearest Neighbours Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred_KNN = classifier.predict(X_test)

#Confusion Matrix for KNN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_KNN)

#Accuracy for KNN
from sklearn.metrics import accuracy_score
ac_score_KNN = accuracy_score(y_test,y_pred_KNN)

#Precision for KNN
from sklearn.metrics import precision_score
p_score_KNN = precision_score(y_test,y_pred_KNN)

#Recall for KNN
from sklearn.metrics import recall_score
r_score_KNN = recall_score(y_test,y_pred_KNN)

#F1-Score for KNN
from sklearn.metrics import f1_score
f1_score_KNN = f1_score(y_test,y_pred_KNN)

#ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score_KNN = roc_auc_score(y_test,y_pred_KNN)

#Log Loss
from sklearn.metrics import log_loss
log_loss_KNN = log_loss(y_test,y_pred_KNN)

