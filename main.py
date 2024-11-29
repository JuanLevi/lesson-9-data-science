import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Splitting the data into training and testing
from sklearn.model_selection import train_test_split
# Algorithm that we would be using
from sklearn.tree import DecisionTreeClassifier
# for finding accuracy
from sklearn import metrics


data = pd.read_csv("iris.csv")

print(data.info())
print(data.describe())



data["species"]=data["species"].replace({"setosa":0,"versicolor":1,"virginica":2,})
print(data["species"])

plt.subplot(221)
plt.scatter(data["sepal_length"], data["species"], s=10, c="red", marker="o")
plt.xlabel("Sepal length")
plt.ylabel("Species")

plt.subplot(222)
plt.scatter(data["sepal_width"], data["species"], s=10, c="blue", marker="o")
plt.xlabel("Sepal wdith")
plt.ylabel("Species")

plt.subplot(223)
plt.scatter(data["petal_length"], data["species"], s=10, c="green", marker="o")
plt.xlabel("Petal length")
plt.ylabel("Species")

plt.subplot(224)
plt.scatter(data["petal_width"], data["species"], s=10, c="black", marker="o")
plt.xlabel("Petal width")
plt.ylabel("Species")

plt.show()


x=data.iloc[:,:4].values
print(x)


x=data.drop("species",axis=1)
print(x)

y=data["species"]
print(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape)
print(x_test.shape)


model=DecisionTreeClassifier(max_depth=3,random_state=42)

model.fit(x_train,y_train)
predi=model.predict(x_test)

accuracy=metrics.accuracy_score(predi,y_test)
print(accuracy)