////////////////////////////////////////////P1/////////////////////////////////////////////////////////

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from math import sqrt

def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2            
    return sqrt(distance)

def Get_Neighbors(train, test_row, num):
    distance = list() 
    data = []
    for i in train:
        dist = Euclidean_distance(test_row, i)
        distance.append(dist)
        data.append(i)
    distance = np.array(distance)
    data = np.array(data)
    """ we are finding index of min distance """
    index_dist = distance.argsort()
    """ we arange our data acco. to index """
    data = data[index_dist]
    """ we are slicing num number of datas """
    neighbors = data[:num]    
    return neighbors

def predict_classification(train, test_row, num):
    Neighbors = Get_Neighbors(train, test_row, num)
    Classes = []
    for i in Neighbors:
        Classes.append(i[-1])
    prediction = max(Classes, key= Classes.count)
    return prediction

iris = load_iris()

y_iris = iris.target
data = iris.data

data = np.insert(data, 4, y_iris, axis =1)
 
train, test = train_test_split(data, test_size = 0.25)

y_pred = []
y_true = test[:, -1]

for i in test:
    prediction = predict_classification(train, i, 10)   
    y_pred.append(prediction)

def Evaluate(y_true, y_pred):
    n_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            n_correct += 1
    acc = n_correct/len(y_true)
    return acc

print("Accuracy : " , Evaluate(y_true, y_pred))


//////////////////////////////////////////////////P2///////////////////////////////////////////////////////////////////

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30)
knn = KNeighborsClassifier(n_neighbors=2,weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("accuracy :" , accuracy_score(y_test,y_pred))


//////////////////////////////////////////////////P3///////////////////////////////////////////////////////////////////

from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

iris=load_iris()
plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()


//////////////////////////////////////////////////P4///////////////////////////////////////////////////////////////////


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X=np.array([2,4,5,3,2,5,4,3,2,3,4,8,9,6,2])
y=np.array([5,3,5,6,4,7,8,6,5,3,2,6,9,4,3])
theta = [0,0]
def hypothesis(theta, X):
    return theta[0] + theta[1]*X
def cost_calc(theta, X, y):
    return (1/2*m) * np.sum((hypothesis(theta, X) - y)**2)

m = len(X)
def gradient_descent(theta, X, y, epoch, alpha):
    cost = []
    i = 0
    while i < epoch:
        hx = hypothesis(theta, X)
        theta[0] -= alpha*(sum(hx-y)/m)
        theta[1] -= (alpha * np.sum((hx - y) * X))/m
        cost.append(cost_calc(theta, X, y))
        i += 1
    return theta, cost

def predict(theta, X, y, epoch, alpha):
    theta, cost = gradient_descent(theta, X, y, epoch, alpha)
    return hypothesis(theta, X), cost, theta

y_predict, cost, theta = predict(theta,X,y,10, 0.01)

plt.figure()
plt.scatter(X,y, label = 'Original y')
plt.scatter(y, y_predict, label = 'predicted y')
plt.legend(loc = "upper left")
plt.xlabel("input feature")
plt.ylabel("Original and Predicted Output")
plt.show()


///////////////////////////////////////////////////////P5////////////////////////////////////////////////

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load Dataset
df = pd.read_csv('BreastCancerDataset.csv')

print(df.info())
print(df.head(5))
print(df.describe())

# Decide Dependent & Independent Attributes
X = df[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']]
y = df[['diagnosis']]


# Split Train & Test Dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

print("Training Features : ")
print(X_train)
print("Training Labels : ")
print(y_train)
print("Testing Features : ")
print(X_test)
print("Testing Labels : ")
print(y_test)


model = KNeighborsClassifier(n_neighbors = 12,weights = 'distance',metric='euclidean')
model.fit(X_train,y_train)
dataClass = model.predict(X_test)

print("Accuracy_score : ")
print(accuracy_score(y_test,dataClass,normalize=True,sample_weight=None))

print("Confusion_matrix : ")
print(confusion_matrix(y_test,dataClass))



//////////////////////////////////////////////////////P6/////////////////////////////////////////////////

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"%(X_test.shape[0], (y_test != y_pred).sum()))



///////////////////////////////////////////////////////P7////////////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X,y=load_iris(return_X_y=True)

X = X[:, np.newaxis, 2]

# Split the data into training/testing sets
X_train = X[:-75]
X_test = X[-75:]

# Split the targets into training/testing sets
y_train = y[:-75]
y_test = y[-75:]

regr = LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)

print(y_pred)
print(X_test.size)
print(y_test.size)

plt.scatter(X_test,y_test,color="black")
plt.plot(X_test,y_pred, color="blue", linewidth=3)
plt.show()


////////////////////////////////////P8////////////////////////////////////////////////

import pandas
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300, 1300]])

print("Predicted CO2 for(weight:2300,Volume:1300): " , predictedCO2)

print("Coefficient : " , regr.coef_)


///////////////////////////////////////P9/////////////////////////////////////////////

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
df = pd.read_csv("china_gdp.csv")
df.head(10)

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


/////////////////////////////////////////////P10//////////////////////////////////////

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = load_digits()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, random_state=4)

NN = MLPClassifier()

NN.fit(x_train, y_train)

y_pred = NN.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)*100

confusion_mat = confusion_matrix(y_test,y_pred)

print("Accuracy for Neural Network is:",accuracy)
print("Confusion Matrix")
print(confusion_mat)


/////////////////////////////////////////////P11//////////////////////////////////////////////


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

df = load_iris()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=40)


dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

dtree.fit(X_train, y_train)

pred_train_tree= dtree.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_tree)))
print(r2_score(y_train, pred_train_tree))

# Code lines 4 to 6
pred_test_tree= dtree.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_tree))) 
print(r2_score(y_test, pred_test_tree))


dtree1 = DecisionTreeRegressor(max_depth=2)
dtree2 = DecisionTreeRegressor(max_depth=5)
dtree1.fit(X_train, y_train)
dtree2.fit(X_train, y_train)

# Code Lines 5 to 6: Predict on training data
tr1 = dtree1.predict(X_train)
tr2 = dtree2.predict(X_train) 

#Code Lines 7 to 8: Predict on testing data
y1 = dtree1.predict(X_test)
y2 = dtree2.predict(X_test) 


print(np.sqrt(mean_squared_error(y_train,tr1))) 
print(r2_score(y_train, tr1))

# Print RMSE and R-squared value for regression tree 'dtree1' on testing data
print(np.sqrt(mean_squared_error(y_test,y1))) 
print(r2_score(y_test, y1)) 


print(np.sqrt(mean_squared_error(y_train,tr2))) 
print(r2_score(y_train, tr2))

# Print RMSE and R-squared value for regression tree 'dtree2' on testing data
print(np.sqrt(mean_squared_error(y_test,y2))) 
print(r2_score(y_test, y2))

plt.scatter(y_test,tr2,color="black")
plt.plot(y_test,tr2,color="blue")
plt.show()


/////////////////////////////////////////P13///////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

iris=load_iris()


x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()


//////////////////////////////////P14///////////////////////////////////////////////////////////

# Program of KMeans Clustering on Iris Dataset

from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
err=[]
for i in range(1,11):
     est = KMeans(n_clusters=i, n_init=25, init="k-means++", random_state=0)
     Tpred_y=est.fit_predict(X)
     err.append(est.inertia_)

plt.plot(range(1,11), err)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Error')
plt.show()

Fest = KMeans(n_clusters=3, n_init=25, init="k-means++", random_state=0,tol=1e-06)
pred_y=Fest.fit_predict(X)

plt.scatter(X[pred_y == 0,0],X[pred_y == 0,1],s=100,c="red", label="Iris-Sentosa")
plt.scatter(X[pred_y == 1,0],X[pred_y == 1,1],s=100,c="blue", label="Iris-Versicolour")
plt.scatter(X[pred_y == 2,0],X[pred_y == 2,1],s=100,c="green", label="Iris-Verginica")
plt.scatter(Fest.cluster_centers_[:,0], Fest.cluster_centers_[:,1],s=100,c="yellow", label="Centroids")

plt.legend()
plt.show()

print("Score=",homogeneity_score(y,pred_y))


print('......Program Ends.........')



