import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel('1553768847_housing.xlsx') 

#reading few rows
data.head()

#input and ouput
x= data.iloc[:,:-1].values
y= data.iloc[:,9].values

#missing values
from sklearn.preprocessing import Imputer
missingVal = Imputer( missing_values="NaN", strategy="mean", axis=0)
x[:,0:8] = missingVal.fit_transform(x[:,0:8])

#categorical data
from sklearn.preprocessing import LabelEncoder
x_label = LabelEncoder()
x[:,8] = x_label.fit_transform(x[:,8])

#x_label.classes_

#test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,
                                                 random_state=0)

#standardize data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)

#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

predictedVal = lr.predict(x_test)
predictedVal2 = lr.predict(x_train)
#checking accuracy
lr.score(x_train,y_train) # 0.6378966554411029
lr.score(x_test,y_test)   # 0.6253939067197227

# rmse
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,predictedVal)) # 69890.58962190864
np.sqrt(mean_squared_error(y_train,predictedVal2)) # 69615.2854305203

#decision tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)

predictedVal3 = dtr.predict(x_test)
predictedVal4 = dtr.predict(x_train)
dtr.score(x_train,y_train) # 1.0
dtr.score(x_test,y_test) # 0.5446154120725679

#rmse
np.sqrt(mean_squared_error(y_test,predictedVal3)) # 77058.48206826876
np.sqrt(mean_squared_error(y_train,predictedVal4)) # 0.0

#random forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)

predictedVal5 = rfr.predict(x_test)
predictedVal6 = rfr.predict(x_train)
rfr.score(x_train,y_train) # 0.9630174595597636
rfr.score(x_test,y_test) # 0.7277729361300703
#rmse
np.sqrt(mean_squared_error(y_test,predictedVal5)) # 59579.511115479356
np.sqrt(mean_squared_error(y_train,predictedVal6)) # 22247.781493357117


#bonus exercise : extracting median income
d1 = data.drop("median_income",axis=1)
d2 = data.drop(d1,axis=1)

x_train2,x_test2,y_train2,y_test2 = train_test_split(d2,y,
                                                     test_size=0.2)

lr2 = LinearRegression()
lr2.fit(x_train2,y_train2)

predictedVal7 = lr2.predict(x_test2)
predictedVal8 = lr2.predict(x_train2)

#visualizing
plt.scatter(y_train,predictedVal8,color='blue',s=5)
plt.scatter(y_test2,predictedVal7,color='red',s=5)
plt.xlabel('Actual median values')
plt.ylabel('Predicted median values')
plt.show()















