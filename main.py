import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#sepal_length,sepal_width,petal_length,petal_width,species
cloums=["sepal_length","sepal_width","petal_length","petal_width","species"]
df = pd.read_csv('iris.csv',names=cloums)

data = df.values
x=data[:,0:4]
y=data[:,4:]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model = SVC()
model.fit(x_train,y_train)

pr=model.predict(x_test)
print(accuracy_score(y_test,pr)*100)

x_new =[[5,2,3,1]]

pr=model.predict(x_new)
print(pr)