#Artificial Neural Networks

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataPreprocessing import X_train, X_test, y_train,y_test

#importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation = 'relu',input_dim=30))

#Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform',activation='relu'))

#Adding the output layer
classifier.add(Dense(output_dim=1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting ANN to the Training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#Making the predictions and evaluating model
y_pred_ANN= classifier.predict(X_test)
y_pred_ANN=y_pred_ANN.round()


#Confusion Matrix For ANN

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_ANN)

#Accuracy for ANN
from sklearn.metrics import accuracy_score
ac_score_ANN = accuracy_score(y_test,y_pred_ANN)

#Precision for ANN
from sklearn.metrics import precision_score
p_score_ANN = precision_score(y_test,y_pred_ANN)

#Recall for ANN
from sklearn.metrics import recall_score
r_score_ANN = recall_score(y_test,y_pred_ANN)

#F1-Score for ANN
from sklearn.metrics import f1_score
f1_score_ANN = f1_score(y_test,y_pred_ANN)

#ROC AUC
from sklearn.metrics import roc_auc_score
roc_auc_score_ANN = roc_auc_score(y_test,y_pred_ANN)

#Log Loss
from sklearn.metrics import log_loss
y_pred_ANN = y_pred_ANN.astype(np.float64)
log_loss_ANN = log_loss(y_test,y_pred_ANN)
