"""@author: Aditya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from sklearn.utils import resample

#Load dataset
df=pd.read_csv('framingham.csv')

#####Exploratory data analysis #########
df['firstchd'].value_counts().plot(kind='bar')
#df.groupby('firstchd').size().plot(kind='bar')

### Class Imbalance: Up sampling the minority class
df['firstchd'].value_counts()
#0    1487
#1     128

# Separate majority and minority classes
df_majority = df[df.firstchd==0]
df_minority = df[df.firstchd==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=539,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.firstchd.value_counts()

df.groupby('firstchd').mean() # Potential split points can be noticed for fhd for each variable; Every mean is higher for diease except smoke flag which makes sense)
y=df_upsampled['firstchd']
x=df_upsampled.iloc[:,1:9]
y_unbal=df['firstchd']
x_unbal=df.iloc[:,1:9]

############################ Logistics Regression Feature Selection#######################
from sklearn.feature_selection import RFE # recursive feature elimination based on variable importance
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
#print('Acuracy of the Logistics regression model: {:.4f}'.format(score))

## Check for overfitting do 10 fold cross validation
from sklearn.cross_validation import cross_val_score
fold_cv= cross_val_score(logreg, x_train, y_train, cv=10) 

# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
conf_matrix = confusion_matrix(y_test, predictions)

print("Logistics Regression Model Performance")
print("\n Logistics Regression Classification Report"+ "\n"+ classification_report(y_test, predictions))
#print("10 fold cross validation accuracy of the model: {:.3f}".format((fold_cv.mean())))

## Probabilities of patiet getting a heart diease 
post_prob_lr= logreg.predict_proba(x_test)
 
################################# Decision Trees########################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

#from sklearn.tree import export_graphviz
#from sklearn import tree
#import pydotplus

# Split in train and test 
x_tree_train, x_tree_test, y_tree_train, y_tree_test = train_test_split( x, y, test_size = 0.2, random_state = 1)
 
#######Tree with different split crieteria
tree_gini = DecisionTreeClassifier(criterion = "gini")
tree_entropy = DecisionTreeClassifier(criterion = "entropy")

# Fit tree model and Predict on test data 
dot_data= tree_gini.fit(x_tree_train, y_tree_train)
clf= tree_entropy.fit (x_tree_train, y_tree_train)
tree_prediction = tree_gini.predict(x_tree_test)
y_pred= tree_entropy.predict(x_tree_test)

# Performance Evaluation for Decision Trees
print("\n Gini Index Tree Classification Report"+ "\n" + classification_report(y_tree_test, tree_prediction))
print("\n Gini Index Tree Accuracy: {:.3f}".format(accuracy_score(y_tree_test, tree_prediction))) 
print("\n Gini Index Tree Mean square error: {:.5f}".format(mean_squared_error(y_tree_test, tree_prediction))) 

print("\n Entropy Tree Classification Report"+ "\n" + classification_report(y_tree_test, y_pred))
print("Entropy Tree Accuracy: {:.3f}".format(accuracy_score(y_tree_test, y_pred)))
print("\n Entropy Tree Mean square error: {:.5f}".format(mean_squared_error(y_tree_test, y_pred))) 
## Probabilities of patiet getting a heart diease 
t_post_prob= clf.predict_proba(x_tree_test)

## Fitting Random Forest classification
tree_forest= RandomForestClassifier()
clf_4= tree_forest.fit(x_tree_train, y_tree_train)
clf_4_pred= clf_4.predict(x_tree_test)
print(classification_report(y_tree_test, clf_4_pred))
accuracy_score(y_tree_test, clf_4_pred)
print( np.unique( clf_4_pred ) )


###Comparing Model Results using AUC statistic from ROC###
from sklearn.metrics import roc_curve, auc

preds = post_prob_lr[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

preds_tree= t_post_prob[:,1]
t_fpr, t_tpr, t_threshold = roc_curve(y_tree_test, preds_tree)
t_roc_auc = auc(t_fpr, t_tpr)

preds_t4= clf_4.predict_proba(x_tree_test)[:,1]
t4_fpr, t4_tpr, t4_threshold = roc_curve(y_tree_test, preds_t4)
t4_roc_auc = auc(t4_fpr, t4_tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'Log Reg AUC = %0.2f' % roc_auc)
plt.plot(t_fpr, t_tpr, 'g', label = 'Class prob trees AUC = %0.2f' % t_roc_auc)
plt.plot(fpr, tpr, 'b', label = 'Log Reg AUC = %0.2f' % roc_auc)
plt.plot(t4_fpr, t4_tpr, 'y', label = 'Random Forest AUC = %0.2f' % t4_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
