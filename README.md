# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn
```
## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Murali Krishna S 
RegisterNumber:212223230129
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```


## Output:
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/35f44669-760e-4d00-8b62-20a37679a6da)
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/19f720d4-8bb1-44b9-aeea-ff53adfc3b0e)
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/a53822a4-f8a9-448e-9839-92adbac7c4d9)
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/7f429589-8d37-4d3c-b7ab-c9138c3f2f8c)
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/58546b56-4cc9-4b1f-a0ac-d51841967ade)
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/ce2cd8bd-cb57-4133-a44d-ecc5aea10588)
![image](https://github.com/Murali-Krishna0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149054535/028e501a-2d0a-4305-bfbe-9cad83c642df)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
