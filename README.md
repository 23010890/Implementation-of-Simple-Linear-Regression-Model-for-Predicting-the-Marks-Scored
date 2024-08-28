# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARSHINI S
RegisterNumber:212223110010  
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
```
dataset.info()
```
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
```
```
X_test.shape
```
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
```
plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='')
plt.title('Training Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
```
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='silver')
plt.title('Test Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```


## Output:
![Screenshot 2024-08-28 104002](https://github.com/user-attachments/assets/c7e5d23d-82bc-4671-8153-cb0cfb1423d4)

![Screenshot 2024-08-28 104130](https://github.com/user-attachments/assets/23b06d87-1bd3-4f05-ad0b-4c8b489de054)

![Screenshot 2024-08-28 104424](https://github.com/user-attachments/assets/c27d5b87-9df7-4d1e-ac09-b2884544793f)

![Screenshot 2024-08-28 104449](https://github.com/user-attachments/assets/f1f796a4-e311-4734-93e0-09b1bc290d2d)

![Screenshot 2024-08-28 104501](https://github.com/user-attachments/assets/348333b0-4646-4e66-af2a-b46d0ce2ec54)

![Screenshot 2024-08-28 104510](https://github.com/user-attachments/assets/9fabdd59-9e0c-41cb-94a5-e5695e2688be)

![Screenshot 2024-08-28 104522](https://github.com/user-attachments/assets/7538bb08-2ab0-4ea5-9fab-bca719a33903)

![Screenshot 2024-08-28 104531](https://github.com/user-attachments/assets/42daa354-164d-4df1-9f32-737861a2965a)

![Screenshot 2024-08-28 104546](https://github.com/user-attachments/assets/579a6fbd-64b3-4397-a17b-d82a72f7d174)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
