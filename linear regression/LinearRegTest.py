import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

d = {'day' : ['24.05.2004', '25.05.2004', '26.05.2004', '27.05.2004', \
    '28.05.2004', '29.05.2004', '30.05.2004', '31.05.2004', '01.06.2004', \
        '02.06.2004', '03.06.2004', '04.06.2004'],
     'X' : [2, 1,   3, 4,   5.1, 11, 9, 8,  19,  13, 16, 17], 
     'Y' : [4, 2.5, 5, 8.1, 10, 21, 17, 15, 37, 28, 30, 32]}

dataframe = pd.DataFrame(data=d)
print(dataframe.head())
print("Length of the dataframe is :", len(dataframe))

X = np.array(dataframe['X']) 
y = np.array(dataframe['Y'])

X = preprocessing.scale(X)
y = preprocessing.scale(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


X_train = X_train.reshape(len(X_train),1)
X_test = X_test.reshape(len(X_test),1)


print(X_train)
print(y_train)


plt.plot(X_train, y_train, 'bo')


classifier = LinearRegression().fit(X_train, y_train)
print("Classifier coefficients : ",classifier.coef_)
accuracy = classifier.score(X_train, y_train)
print(accuracy)

print("X test",X_test)
X_predicted = classifier.predict(X_test)
plt.plot(X_test, X_predicted, 'ro')

plt.xlabel("Abssices")
plt.ylabel("Ordonnées")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()


