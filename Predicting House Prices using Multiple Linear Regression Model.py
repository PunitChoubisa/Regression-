#!/usr/bin/env python
# coding: utf-8

# In this dataset we are predicting the house prices(dependent variable) with Independent variables (House Size, Distance, etc).
# As Linear Regression is used to predict the relationship between dependent and independent variables we are using the same for predicting the prices of houses.

# In[ ]:


Assumptions of Linear Regression:
(i)Linear relationship.
(ii)Multivariate normality.
(iii)No or little multicollinearity.
(iv)No auto-correlation.
(v)Homoscedasticity.


# In[1]:


import numpy as np                     #loading the packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


reg=pd.read_excel("C:/Users/choub/Downloads/DS - Assignment Part 1 data set.xlsx")


# In[5]:


reg


# In[7]:


reg.info()   #checking the data types


# In[17]:


reg.drop('Transaction date',axis=1,inplace=True)  #dropping the column Transaction Date as it is irrelevant
reg


# In[19]:


reg.describe()    #getting the mean, minimum and maximum values


# In[18]:


sns.pairplot(reg,height=2)


# In[21]:


reg.corr()       #getting the correlation to check the correlation between variables


# In[22]:


sns.heatmap(reg.corr())         #using heatmap to check the correlation 


# In[23]:


from sklearn.preprocessing import StandardScaler   #importing required libraries for regression
r=StandardScaler().fit(reg)
reg1=r.transform(reg)


# In[24]:


reg2=pd.DataFrame(reg1)
reg2


# In[26]:


X=reg2.iloc[:,:-1].values    #representing the independent variables as X
Y=reg2.iloc[:,7]             #representing the dependent variables as Y


# In[27]:


print(Y)


# In[28]:


print(X)           


# In[83]:


from sklearn.model_selection import train_test_split    #importing the libraries for splitting the data and splitting the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=80,random_state=1)


# In[84]:


from sklearn.linear_model import LinearRegression   #importing the linear regression library and fitting the model
model=LinearRegression()
model.fit(X,Y)


# In[85]:


Y_pred_test=model.predict(X_test)     #fitting the test model
Y_pred_test


# In[86]:


print("Intercept:",model.intercept_)


# In[87]:


print("Coefficients:",model.coef_)


# In[88]:


Y_pred_train=model.predict(X_train)  #fitting the train model
Y_pred_train


# In[89]:


print("Intercept:",model.intercept_)    
print("Coefficients:",model.coef_)


# In[90]:


from sklearn import metrics                     #importing the metrics library for getting MAE and MSE
MAE_train=metrics.mean_absolute_error(Y_train,Y_pred_train)    #MAE value for train and test dataset
MAE_test=metrics.mean_absolute_error(Y_test,Y_pred_test)
print("MAE for train data is: {}".format(MAE_train))
print("MAE for test data is {}".format(MAE_test))


# In[91]:


MSE_train=metrics.mean_squared_error(Y_train,Y_pred_train)   #MSE value for train and test dataset
MSE_test=metrics.mean_squared_error(Y_test,Y_pred_test)
print("MSE for train data is: {}".format(MSE_train))
print("MSE for test data is {}".format(MSE_test))


# In[92]:


RMSE_train=np.sqrt(metrics.mean_squared_error(Y_train,Y_pred_train))   #RMSE value for train and test dataset
RMSE_test=np.sqrt(metrics.mean_squared_error(Y_test,Y_pred_test))
print("RMSE for train data is: {}".format(RMSE_train))
print("RMSE for test data is {}".format(RMSE_test))


# In[93]:


Yhat = model.predict(X_train)                     #Rsq and Adjusted R sq value for train dataset
SS_Residual = sum((Y_train-Yhat)**2)
SS_Total = sum((Y_train-np.mean(Y_train))**2)
R_Square =1-(float(SS_Residual))/SS_Total
Adj_R_Square = 1-(1-R_Square)*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1)


# In[94]:


print(R_Square,Adj_R_Square)


# In[95]:


Y_hat = model.predict(X_test)                 #Rsq and Adjusted R sq value for test dataset
SS_Residual = sum((Y_test-Y_hat)**2)
SS_Total = sum((Y_test-np.mean(Y_test))**2)
R_Square =1-(float(SS_Residual))/SS_Total
Adj_R_Square = 1-(1-R_Square)*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1)


# In[75]:


print(R_Square,Adj_R_Square)


# The R-square value for training dataset is 62% and Adjusted R-square value is 58%.
# The R-square value for test dataset is 56% and Adjusted R-Square value for test dataset is 51%.
# Hence we can say from the above result we can conclude that the house prices can be predicted by the independent variables as the R square value is more than 50%. Though the value is lesser the prices can be predicted and hence the regression method can be used for predicting the values.
# 
