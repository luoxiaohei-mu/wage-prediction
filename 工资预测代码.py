import numpy as np
import pandas  as pd  
from sklearn import datasets, linear_model
#导入数据
nslw=pd.read_csv('c://nslw//nslw88.csv')
print(nslw.head(5))
print(nslw.shape)
print(nslw.describe())
#选择回归参照race,grade,collgrad,south,smsa,occupation,hours,ttl_exp,tenure
X=nslw[['race','grade','collgrad','south','smsa','occupation','hours','ttl_exp','tenure']]
print(X.head(5))
y=nslw[['wage']]
print(y.head(5))
#划分训练集和测试集
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#进行线性回归模型
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)
#模型拟合测试集
y_pred=linreg.predict(X_test)
from sklearn import metrics
print("MSE",metrics.mean_squared_error(y_test, y_pred))
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#更换回归参照
X=nslw[['grade','collgrad','south','smsa','occupation','hours','ttl_exp']]
y=nslw[['wage']]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
#模型拟合测试集
y_pred=linreg.predict(X_test)
from sklearn import metrics
print(linreg.intercept_)
print(linreg.coef_)
print("MSE",metrics.mean_squared_error(y_test, y_pred))
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#结论：第二个模型的MSE值更小，因此第二个模型的拟合优度更高



