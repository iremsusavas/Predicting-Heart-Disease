#Support Vector Classifier

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataPreprocessing import X_train, X_test, y_train,y_test


#Support Vector Classifier
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear' , random_state=0) #for linear kernel
#to use RBF kernel run the given code -> classifier = SVC(kernel = 'rbf' , random_state=0) 
#to use sigmoid kernel run the given code -> classifier = SVC(kernel = 'sigmoid' , random_state=0)
#classifier = SVC(kernel = 'sigmoid' , random_state=0)
classifier.fit(X_train, y_train)
y_pred_SVC = classifier.predict(X_test)

#Confusion Matrix for SVC
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_SVC)

#Accuracy for SVC
from sklearn.metrics import accuracy_score
ac_score_SVC = accuracy_score(y_test,y_pred_SVC)

#Precision for SVC
from sklearn.metrics import precision_score
p_score_SVC = precision_score(y_test,y_pred_SVC)

#Recall for SVC
from sklearn.metrics import recall_score
r_score_SVC = recall_score(y_test,y_pred_SVC)

#F1-Score for SVC
from sklearn.metrics import f1_score
f1_score_SVC = f1_score(y_test,y_pred_SVC)

#ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score_SVC = roc_auc_score(y_test,y_pred_SVC)

#Log Loss
from sklearn.metrics import log_loss
log_loss_SVC = log_loss(y_test,y_pred_SVC,eps=1e-15,normalize=True)