# Coronary Heart Diease
## **Objective**
The Objective of this project is to predict probability of patient suffering from a coronary heart diease based on his/her medical records. We take a supervised learning approache namely Logistics Regression and Decision trees to predict the probability and classify the possible patients. 

## **Data**
The Data for this project is borrowed from the [framigham heart study](https://en.wikipedia.org/wiki/Framingham_Heart_Study). The data can be found in _framingham.csv_. The variables in the data are explained below 
* AGE: age at exam 2 
* SBP21: first systolic blood pressure at exam 2
* BP22: second systolic blood pressure at exam 2
* SBP31:  first systolic blood pressure at exam 3
* SBP32: second systolic blood pressure at exam 3
* SMOKE:  present smoking at exam 1
* CHOLEST2: serum cholesterol at exam 2
* CHOLEST2: serum cholesterol at exam 3
* firstCHD: indicator of heart diease at exam 3-6 based on following variables 

## *Methodology*

### _Exploratory Data Anaylsis : Class Imbalance Problem_
A patient suffering from a heart diease is a rare case. This is pretty evident when we explore the dataset. In order to address the class imbalance problems we  _upsample the minority class_ in our case the positive class with a final representation of classes in a 2:1 ratio of non-diease to diease indicators. 

### Approach 1: _Logistic Regression with feature selection_
Logistic Regression is used to classify the patients and then the  logit values are used to calculate the posterior probability namely the probability of getting a first order heart diease. Feature Selection based on varaible importance is done before a model is fit to the data. 

### Approach 2: _Decision Tree Model ( Class probility tee and Random forest)_ 
A class probability Decision tree designed to minimize the average square error and a random forest model is deisgned. Trees using both Gini index and entropy split criteria are developed for comparison. The tree splits the data into nodes which result in decision rules classifying whether the patient has heart diease or not.

Due to class imbalance problem accuracy is not a good measure and hence both the approaches are evaluated on their AUC measure from the ROC curve. Random forest does the best job at classifying the test data.
