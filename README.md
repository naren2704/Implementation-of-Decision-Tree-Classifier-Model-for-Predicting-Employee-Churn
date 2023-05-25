### Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
 1.Hardware – PCs
 2.Anaconda – Python 3.7 Installation / Jupyter notebook
 3.Algorithm
 4.import the required libraries.
 5.Upload and read the dataset.
 6.Check for any null values using the isnull() function.
 7.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
 8.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NARENDRAN B 
RegisterNumber:  212222240069
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```
## Output:
## data.head()
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/23683ccc-de8e-43a9-921a-0fed0d198512)
## data.info()
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/18c09de5-9bb4-4312-b8d1-3a7b070fd7c7)
## isnull() and sum()
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/d8c95f23-47d5-43db-88f9-de7b17c0667a)
## data value counts()
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/07fc6a78-40eb-40ef-978f-17a0381b7b4a)
## data.head() for salary
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/506651dc-1ebf-4518-8142-c2ff46e9810d)
## x.head()
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/ce5be054-8020-4e22-9e29-a6c809a348f9)
## accuracy value
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/af66b379-71a3-465b-b148-0c38f9685e78)
## data prediction
![image](https://github.com/naren2704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118706984/34e73a94-8769-4aae-a818-d3a01caa80de)
## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
