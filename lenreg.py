import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"production.csv")
data = data.drop_duplicates(keep=False, inplace=False)
#data = (data - data.mean()) / data.std()

X = data['Rainfall'].values
Y = data['Production'].values

#split the dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

length = len(X_train)
x_mean = np.mean(X_train)
y_mean = np.mean(Y_train)

numerator = 0
denominator = 0

for i in range(length):
   numerator += ((X_train[i] - x_mean) * (Y_train[i] - y_mean))
   denominator += (X_train[i] - x_mean) ** 2

slope = numerator / denominator
intercept = (y_mean - (slope * x_mean))
print('SLOPE : ' + str(slope) + '\nINTERCEPT : ' + str(intercept))

rmse = 0
for i in range(len(X_test)):
   y_predicted = slope * X_test[i] + intercept
   rmse += (Y_test[i] - y_predicted) ** 2

rmse = np.sqrt(rmse / length)
print('RMSE : ' + str(rmse))
x_max = np.max(X)
x_min = np.min(X)

x_plot = np.linspace(x_max, x_min, length)
y_plot = (slope * x_plot) + intercept

plt.xlabel('RainFall')
plt.ylabel('Yield')
plt.legend()
plt.plot(x_plot, y_plot, color='#F4A950', label='Regression Line')
plt.scatter(X, Y, color='#161B21', label='Scatter Plot')

#print("Accuracy:",regressor.score(Y_test, Y_predicted))

#a=int(input())
#b=a*slope+intercept
#print(b)
#plt.scatter(a, b, color='#0000FF', label='Scatter Plot')

plt.legend()
plt.show()


