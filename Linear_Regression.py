#!/usr/bin/env python
# coding: utf-8

# In[157]:


import numpy as np


# In[158]:


import pandas as pd


# In[236]:


class linear_regression(object):
    def fit(self,x_train,y_train):
        self.X = np.array(x_train)
        self.m = np.size(self.X,axis = 0)
        self.n = np.size(self.X,axis = 1)
        self.X = np.hstack((np.ones((self.m,1)),self.X))
        self.y = np.array(y_train)
        self.theta = np.random.random((self.n+1,1))
        self.hypothesis = np.dot(self.X,self.theta)
        self.cost_func = sum((self.hypothesis-self.y)**2)/self.m
    def get_params(self):
        self.alpha = 0.0001
        self.iter_count = 0
        while self.iter_count<3000:
            self.iter_count+=1
            self.theta = self.theta - (self.alpha/self.m)*(np.dot(self.X.T,(self.hypothesis-self.y)))
        return self.theta
    def predict(self,x_test):
        self.x_test = np.hstack((np.ones((np.size(x_test,axis=0),1)),x_test))
        self.prediction = np.dot(self.x_test,self.get_params())
        return self.prediction
    def score(self,x_test,y_test):
        self.u = ((y_test-self.predict(x_test))**2).sum()
        self.v = ((y_test-np.mean(y_test))**2).sum()
        return (1-(self.u/self.v))


# In[237]:


df = pd.read_csv("student-mat.csv",sep = ";",header=None)
y = df[32]
x = df[[2,6,7,23,24,25,26,27,28,29,30,31]]


# In[238]:


x.drop(0,axis=0,inplace=True)
x.head()
x.info()


# In[239]:


y.drop(0,axis=0,inplace=True)
y.head()


# In[240]:


x.shape


# In[241]:


model = linear_regression()


# In[242]:


model.fit(x,y)


# In[212]:


model.get_params()


# In[213]:


model.predict(X)


# In[92]:


model.score(X,y)


# In[ ]:




