#!/usr/bin/env python
# coding: utf-8

# ### Prediction using Decision Tree Algorithm
# ### Usama Norat

# In[195]:


import numpy as np 
import pandas as pd
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## load the given dataset

# In[83]:


cd C:\Users\Usama Norat\OneDrive\Desktop\DATA SCIENCE\Spark Data


# In[84]:


df=pd.read_csv("Iris.csv")


# In[85]:


df.head(10)


# ## describe the data

# In[86]:


df.describe()


# ### Data Shape  

# In[87]:


df.shape


# ### print all the column names  

# In[88]:


df.columns


# ### Display unique value in Species Column

# In[89]:


df.Species.unique()


# ### Data Types

# In[90]:


df.dtypes


# ### Check Null value in any column

# In[91]:


print(df.isnull().sum())


# ### Display Total value in Species Column 

# In[92]:


df["Species"].value_counts()


# ### Apply Lable-encoding on categorical data

# In[93]:


from sklearn.preprocessing import LabelEncoder


# In[94]:


enc =LabelEncoder()


# In[96]:


df['Species'] = enc.fit_transform(df['Species'])


# In[200]:


df.head(25)


# In[98]:


df.shape


# In[99]:


df.info()


# In[109]:


df=df.drop('Id',axis=1)


# ### Create the features and target Data

# In[110]:


x= df.drop("Species", axis=1) 
y= df["Species"]


# ### Perform scaling on features data

# In[111]:


from sklearn.preprocessing import StandardScaler


# In[112]:


# Normalize Features
scaler = StandardScaler()
X_scal = scaler.fit_transform(x)


# ### Split the data in training and testing sets

# In[113]:


from sklearn.model_selection import train_test_split


# In[166]:


np.random.seed(23)
x_train,x_test,y_train,y_test = train_test_split(X_scal,y,test_size=0.3)


# In[157]:


x_train.shape


# In[158]:


x_test.shape


# ### Fit the decision tree model 

# In[163]:


from sklearn.tree import DecisionTreeClassifier


# In[167]:


model = DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[168]:


print("Training Accuracy:" , model.score(x_train,y_train))
print("Testing Accuracy:" , model.score(x_test,y_test))


# ### Now Predict the data 

# In[170]:


y_pred = model.predict(x_test)
y_pred


# In[171]:


comparison = pd.DataFrame(list(zip(y_pred, y_test)), columns=["Predicted","Actual"])


# In[172]:


comparison[:15]


# ### Check the Accuracy score, Confusion matrix 

# In[177]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,f1_score


# In[178]:


accuracy_score(y_pred,y_test)


# In[180]:


confusion_matrix(y_pred,y_test)


# In[183]:


y_prob = model.predict_proba(x_test)
y_prob


# ### Now visualise Decision Tree

# In[197]:


from sklearn import tree
plt.figure(figsize=(25,20))
tree.plot_tree(model,
              feature_names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],
               class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"], filled=True);


# In[ ]:




