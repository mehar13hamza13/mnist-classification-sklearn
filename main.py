import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("dataset/train.csv")
print(data.columns)
data = data.to_numpy()


classifier = DecisionTreeClassifier()

x_train = data[0:21000, 1:]
y_train = data[0:21000, 0]

classifier.fit(x_train, y_train)


# Now testing the dataset
x_test = data[21000:,1:]
y_test = data[21000:, 0]

d = x_test[8]
d.shape = (28, 28)
pt.imshow(255-d, cmap="gray")


pt.show()

# testing the accuracy
p = classifier.predict(x_test)
count = 0
for i in range(0, 21000):
    count+=1 if p[i] == y_test[i] else 0

print("Accuracy = ", (count/21000)*100)
