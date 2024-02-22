# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import pandas, numpy, matplotlib.pyplot, and LinearRegression from scikit-learn.
2. Read Data: Read 'student_scores.csv' into a DataFrame (df) using pd.read_csv().
3. Data Preparation: Extract 'Hours' (x) and 'Scores' (y). Split data using train_test_split().
4. Model Training: Create regressor instance. Fit model with regressor.fit(x_train, y_train).
5. Prediction: Predict scores (y_pred) using regressor.predict(x_test).
6. Model Evaluation & Visualization: Calculate errors. Plot training and testing data. Print errors.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Praveenkumar S
RegisterNumber: 212222230108

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv("/content/student_scores.csv")
print(df.head())

print(df.tail())

x = df.iloc[:,:-1].values
print(x)

y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 

```

# Output:
## HEAD
![305889410-c12f0ba7-7e54-49f8-bbab-d4798be56610](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/c115df3d-f117-49ba-830c-b06a121995f8)

## TAIL
![305889536-3b4351f4-8ac0-4eaf-9b30-09ce33dcc5a0](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/4e2bfe00-267e-44da-a9d1-a12d8c1c9cb8)

## X and Y values
![305889617-24181ef0-f820-4628-83a9-6f38dd3f4a07](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/52c4fb75-2ee5-4e91-94c7-db6bc61491ba)

## Prediction of X and Y
![305889738-d516f109-5227-4fd1-9f6c-ae023db135a2](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/23e73859-a5da-4354-8bf5-0ce0dbac10cc)

## MSS,MSE and RMSE
![305890203-1f372d2e-e4ae-4c7a-8926-5dc9ab3a8cdb](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/b68ce3fe-410d-4e93-ba4a-5ea872ecf5cc)

## Training Set
![305890337-cd85f04a-85a0-4104-bad6-07cf03ab097f](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/e667bc3b-d584-48e3-8f33-513e69c11b1a)

## Testing Set
![305890401-7af30789-2b3c-4674-a0c1-10e3aca1205b](https://github.com/Praveenkumar2004-dev/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559827/d00c2bbf-a8c2-4d7c-87a0-ba108ddf5f82)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
