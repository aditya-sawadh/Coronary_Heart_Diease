# Coronary_Heart_Diease
This project predicts the probability of a patient suffering from first order coronary heart diease based on his/her stats during prior visits
 
The variables in the data are explained below 
AGE = age at exam 2, 
SBP21 = first systolic blood pressure at exam 2,
SBP22 = second systolic blood pressure at exam 2,
SBP31 = first systolic blood pressure at exam 3,
SBP32 = second systolic blood pressure at exam 3,
SMOKE = present smoking at exam 1,
CHOLEST2= serum cholesterol at exam 2,
CHOLEST2= serum cholesterol at exam 3,
firstCHD = indicator CHD at exam 3-6 based on following variables 

The data is divided into training and and test data sets.

Approach 1: Logistic Regression
Logistic Regression is used to classify the patients and then use logit value to calculate the posterior probability namely the probability of getting a first order heart diease. Feature Selection is done before a model is fit to the data. 

Approach 2: Decision tree 
A class probability Decision tree is designed to minimize the average square error. Trees using both Gini index and entropy split criteria are developed for comparison. The tree splis data into nodes which result in decision rules classifying whether the patient has heart diease or not.

Both the approaches are evaluated on their performance parameters Accuracy, Precision, Recall and F1 measure 
