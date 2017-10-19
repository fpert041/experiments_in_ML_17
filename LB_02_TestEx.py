#PRESS <Ctrl>+<Enter> to execute this cell

#%matplotlib inline

#In this cell, we load the iris/flower dataset we talked about in class
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# view a description of the dataset
print(iris.DESCR)

%matplotlib inline
#above: directive to plot inline


#PRESS <Ctrl>+<Enter> to execute this cell

#This populates info regarding the dataset.  Amongst others, we can see that the 'features' used are sepal length and width and petal length and width
#Lets plot sepal length against sepal width, using the target labels (which flower)
X=iris.data
Y=iris.target

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

#first two features are sepal length and sepal width
plt.show()

%matplotlib inline

#here's also how to plot in 3d:
from mpl_toolkits.mplot3d import Axes3D #

#create a new figure
fig = plt.figure(figsize=(5,5))


#this creates a 1x1 grid (just one figure), and now we are plotting subfigure 1 (this is what 111 means)
ax = fig.add_subplot(111, projection='3d')

#plot first three features in a 3d Plot.  Using : means that we take all elements in the correspond array dimension
ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=Y)
