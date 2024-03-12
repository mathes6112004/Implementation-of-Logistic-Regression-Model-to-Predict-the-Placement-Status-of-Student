# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/dc010912-ff28-437e-b329-67173c99866a)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/acffd571-4113-4fa4-8dc6-99753866bb2a)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/94292436-44d4-41df-85d2-3d6be903f493)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/7c9d1414-6164-4bca-a2e6-7f8f98a26f56)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/91d67c23-ffc2-42eb-8eae-41827cc3bc89)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/37d8a87b-2119-435b-b3a8-206455c527cf)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/a781beb7-6517-4844-b7c5-b8fae0aafe07)
![image](https://github.com/mathes6112004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477782/c76f0b27-f7ef-4844-bb44-13e2a978ce6d)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
