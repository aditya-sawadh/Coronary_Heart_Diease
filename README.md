# Coronary_Heart_Diease
This project predicts the probability of a patient suffering form first order coronary heart diease (indicator CHD) at exam 3-6 based on following variables 

AGE = age at exam 2
SBP21 = first systolic blood pressure at exam 2
SBP22 = second systolic blood pressure at exam 2
SBP31 = first systolic blood pressure at exam 3
SBP32 = second systolic blood pressure at exam 3
SMOKE = present smoking at exam 1
serum cholesterol at exam 2
serum cholesterol at exam 3

Approach 1: Logistic Regression
Logistic Regression is used to classify the patients and then use logit value to calculate the posterior probability namely the probability of getting a first order heart diease 

Approach 2: Decision tree splits the data into nodes which result in decision rules classifying whether the patient has heart diease or not.
