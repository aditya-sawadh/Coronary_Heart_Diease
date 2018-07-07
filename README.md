# Coronary Heart Diease
## **Objective**
The Objective of this project is to predict what is the probability of patient will suffer from a coronary heart diease in future based on his/her medical records. We take a supervised learning approach ( Logistics Regression and Decision trees) to predict and identify these patients. 

## **Data**
The Data for this project is borrowed from the famous [framigham heart study](https://en.wikipedia.org/wiki/Framingham_Heart_Study). The data can be found at _framingham.csv_. The variables in the data are explained below 
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

### _Exploratory Data Anaylsis_

### Approach 1: _Logistic Regression with feature selection_
Logistic Regression is used to classify the patients and then use logit value to calculate the posterior probability namely the probability of getting a first order heart diease. Feature Selection is done before a model is fit to the data. 

### Approach 2: _Decision Tree Model_ 
A class probability Decision tree is designed to minimize the average square error. Trees using both Gini index and entropy split criteria are developed for comparison. The tree splis data into nodes which result in decision rules classifying whether the patient has heart diease or not.

Both the approaches are evaluated on their performance parameters Accuracy, Precision, Recall and F1 measure 
