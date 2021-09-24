#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


#reding file
customer_churn = pd.read_csv("D:\\data\\customer_churn.csv") 


# In[8]:


#finding the first few rows
customer_churn.head()


# In[9]:


#Extracting 5th column
customer_5=customer_churn.iloc[:,4] 
customer_5.head()


# In[10]:


#Extracting 15th column
customer_15=customer_churn.iloc[:,14] 
customer_15.head()


# In[12]:


#'Extracting male senior citizen with payment method-> electronic check'
senior_male_electronic=customer_churn[(customer_churn['gender']=='Male') & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head()


# In[13]:


#tenure>70 or monthly charges>100
customer_total_tenure=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head()


# In[14]:


#cotract is 'two year', payment method is 'Mailed Check', Churn is 'Yes'
two_mail_yes=customer_total_tenure=customer_churn[(customer_churn['Contract']=='Two year') & (customer_churn['PaymentMethod']=='Mailed check') & (customer_churn['Churn']=='Yes')]
two_mail_yes


# In[15]:


#Extracting 333 random records
customer_333=customer_churn.sample(n=333)
customer_333.head()


# In[16]:


len(customer_333)


# In[17]:


#count of levels of churn column
customer_churn['Churn'].value_counts()


# In[12]:


#-------------------------------Data Visualization------------------#


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


#bar-plot for 'InternetService' column
plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),customer_churn['InternetService'].value_counts().tolist(),color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of categories')
plt.title('Distribution of Internet Service')


# In[20]:


#histogram for 'tenure' column
plt.hist(customer_churn['tenure'],color='green',bins=30)
plt.title('Distribution of tenure')


# In[21]:


#scatterplot 
plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'],color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')


# In[22]:


#Box-plot
customer_churn.boxplot(column='tenure',by=['Contract'])


# In[23]:


#-----------------------Linear Regresssion----------------------


# In[24]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[25]:


x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['MonthlyCharges']


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[27]:


#building the model
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(x_train,y_train)


# In[28]:


#predicting the values
y_pred = simpleLinearRegression.predict(x_test)


# In[29]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse


# In[30]:


#----------------------------------Logistic Regression-------------------------------


# In[34]:


x=pd.DataFrame(customer_churn['MonthlyCharges'])
y=customer_churn['Churn']


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65,random_state=0)


# In[36]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[37]:


y_pred = logmodel.predict(x_test)


# In[38]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test)


# In[39]:


#--------------Multiple logistic regression-------------------


# In[40]:


x=pd.DataFrame(customer_churn.loc[:,['MonthlyCharges','tenure']])
y=customer_churn['Churn']


# In[41]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)


# In[42]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[43]:


y_pred = logmodel.predict(x_test)


# In[45]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[46]:


#---------------decision tree---------------


# In[47]:


x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['Churn']


# In[48]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[49]:


from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(x_train, y_train)  


# In[50]:


y_pred = classifier.predict(x_test)  


# In[51]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))   
print(accuracy_score(y_test, y_pred))  


# In[52]:


#--------------random forest---------------------


# In[53]:


x=customer_churn[['tenure','MonthlyCharges']]
y=customer_churn['Churn']


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[56]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)


# In[57]:


y_pred=clf.predict(x_test)


# In[58]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




