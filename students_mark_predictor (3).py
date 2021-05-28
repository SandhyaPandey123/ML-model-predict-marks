#!/usr/bin/env python
# coding: utf-8

# # Business Problem

# In[ ]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Load Dataset

# In[ ]:


path = r"https://drive.google.com/uc?export=download&id=13ZTYmL3E8S0nz-UKl4aaTZJaI3DVBGHM"
df  = pd.read_csv(path)


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# ## Discover and visualize the data to gain insights

# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


plt.scatter(x =df.study_hours, y = df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()


# ## Prepare the data for Machine Learning algorithms 

# In[ ]:


# Data Cleaning


# In[10]:


df.isnull().sum()


# In[11]:


df.mean()


# In[ ]:


df2 = df.fillna(df.mean())


# In[13]:


df2.isnull().sum()


# In[14]:


df2.head()


# In[ ]:


# split dataset


# In[16]:


X = df2.drop("student_marks", axis = "columns")
y = df2.drop("study_hours", axis = "columns")
print("shape of X = ", X.shape)
print("shape of y = ", y.shape)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=51)
print("shape of X_train = ", X_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of X_test = ", X_test.shape)
print("shape of y_test = ", y_test.shape)


# # Select a model and train it

# In[ ]:


# y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[19]:


lr.fit(X_train,y_train)


# In[20]:


lr.coef_


# In[21]:


lr.intercept_


# In[22]:


m = 3.93
c = 50.44
y  = m * 4 + c 
y


# In[23]:


lr.predict([[4]])[0][0].round(2)


# In[24]:


y_pred  = lr.predict(X_test)
y_pred


# In[25]:


pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])


# ## Fine tune our model
# 

# In[26]:


lr.score(X_test,y_test)


# In[27]:


plt.scatter(X_train,y_train)


# In[28]:


plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = "r")


# ## Present our solution

# ## Save Ml Model

# In[29]:


import joblib
joblib.dump(lr, "student_mark_predictor.pkl")


# In[ ]:


model = joblib.load("student_mark_predictor.pkl")


# In[31]:


model.predict([[5]])[0][0]

