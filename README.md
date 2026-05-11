# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SHAKTHI BALAN V
RegisterNumber: 212225230259
*/
```
```


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("50_Startups.csv")
X_data = data.iloc[:,0].values.astype(float)  
y_data = data.iloc[:,4].values.astype(float)  
plt.figure(figsize=(6,4))
plt.scatter(X_data, y_data)
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")
def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    square_err = (h - y) ** 2
    return (1 / (2 * m)) * np.sum(square_err)
m = len(X_data)
X = np.append(np.ones((m,1)),
              X_data.reshape(m,1),
              axis=1)
y = y_data.reshape(m,1)
theta = np.zeros((2,1))
print(computeCost(X, y, theta))
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * (1/m) * error
        theta = theta - descent
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
theta, J_history = gradientDescent(
    X, y, theta, 0.0000000000001, 1500)
print("h(x) =" + str(round(theta[0,0],2)) +
      " + " + str(round(theta[1,0],6)) + "x1")
plt.figure(figsize=(6,4))
plt.plot(J_history[0:50])
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function using Gradient Descent")
plt.figure(figsize=(6,4))
plt.scatter(X_data, y_data)
x_value = np.array(X_data)
y_value = theta[0,0] + theta[1,0] * x_value
plt.plot(x_value, y_value, color='red')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")
plt.show()


```

## Output:
<img width="751" height="483" alt="image" src="https://github.com/user-attachments/assets/85c2d439-f76e-4af0-9424-d993ce509a84" />
<img width="743" height="448" alt="image" src="https://github.com/user-attachments/assets/bc2d9347-7f56-4643-9c6d-13e1dee22353" />
<img width="697" height="472" alt="image" src="https://github.com/user-attachments/assets/b7f9a317-5aab-4ff9-a752-76b8b260c473" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
