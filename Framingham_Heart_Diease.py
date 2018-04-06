## Author: Aditya Sawadh 
#################
## Import the libaries required
#################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#Load dataset
framingham=pd.read_csv('framingham.csv')

#Exploratory data analysis 
framingham['firstchd'].value_counts().plot(kind='bar')
framingham.groupby('firstchd').mean() # Potential split points can be noticed for fhd for each variable; Every mean is higher for diease except smoke flag which makes sense)
framingham.hist()
y=framingham['firstchd']
x=framingham.iloc[:,1:9]

############################ Logistics Regression Feature Selection#######################
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression() #instance of model
rfe = RFE(logreg)
rfe = rfe.fit(x, y)
print(rfe.support_) # Selected features
print(rfe.ranking_)
#subset the selected features 
sel_feat = ['age','sbp32','smoke','cholest2']
X=x[sel_feat]

#Model Parameters 
import statsmodels.api as sm
logit=sm.Logit(y,X)
result=logit.fit()
print(result.summary())

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test= train_test_split(X,y,random_state=0,train_size=0.8)
logreg.fit(x_train,y_train)# Fit the model
predictions= logreg.predict(x_test) # Make predctions on test data

###Model Evaluation
#Accuracy
score = logreg.score(x_test, y_test)
print('Acuracy of the Logistics regression model: {:.4f}'.format(score))

## Check for overfitting do 10 fold cross validation
from sklearn.cross_validation import cross_val_score
fold_cv= cross_val_score(logreg, x_train, y_train, cv=10) 

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confusion_matrix = confusion_matrix(y_test, predictions)

print("Logistics Regression Model Performance")
print('Acuracy of the Logistics regression model: {:.4f}'.format(score))
print("\n Logistics Regression Classification Report"+ "\n"+ classification_report(y_test, predictions))
print("10 fold cross validation accuracy of the model: {:.3f}".format((fold_cv.mean())))

## Probabilities of patiet getting a heart diease 
posterior_probabilites= print(logreg.predict_proba(x_test))
 
################################# Decision Trees########################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
#from sklearn.tree import export_graphviz
#from sklearn import tree
#import pydotplus

# Split in train and test 
x_tree_train, x_tree_test, y_tree_train, y_tree_test = train_test_split( x, y, test_size = 0.2, random_state = 1)
 
#######Tree with different split crieteria
tree_gini = DecisionTreeClassifier(criterion = "gini")
tree_entropy = DecisionTreeClassifier(criterion = "entropy")

# Fit tree model and Predict on test data 
tree_gini.fit(x_tree_train, y_tree_train)
tree_entropy.fit (x_tree_train, y_tree_train)
tree_prediction = tree_gini.predict(x_tree_test)
y_pred= tree_entropy.predict(x_tree_test)


# Performance Evaluation for Decision Trees
print("\n Gini Index Tree Classification Report"+ "\n" + classification_report(y_test, tree_prediction))
print("\n Gini Index Tree Accuracy: {:.3f}".format(accuracy_score(y_test, tree_prediction))) 
print("\n Gini Index Tree Mean square error: {:.5f}".format(mean_squared_error(y_test, tree_prediction))) 

print("\n Entropy Tree Classification Report"+ "\n" + classification_report(y_test, y_pred))
print("Entropy Tree Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("\n Entropy Tree Mean square error: {:.5f}".format(mean_squared_error(y_test, y_pred))) 


