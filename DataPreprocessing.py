#Data Preprocessing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading the data
dataset = pd.read_csv('heart.csv')


dataset.hist()  #Histogram of the dataset

#Bar Plot For Target Classes
from matplotlib import rcParams
rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')

#No need to search for missing values because the control was done with using weka


#Data Processing and Scaling
from sklearn.preprocessing import StandardScaler
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']) #To work with categorical variables the categorical colunmns are broken into dummy columns
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] #Scaling the numeric variables
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

#Importing the dataset
y = dataset['target']
X = dataset.drop(['target'],axis=1)


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X , y , test_size=0.33 , random_state=0 )


