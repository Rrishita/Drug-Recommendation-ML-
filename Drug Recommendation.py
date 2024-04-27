#!/usr/bin/env python
# coding: utf-8

# # Understanding the dataset
# 

# ### Importing all the necessary libraries

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# ### Reading the dataset

# In[4]:


df = pd.read_csv("drug.csv")
df.head()


# ### Number of rows and columns in the dataset

# In[5]:


df.shape


# ### Info of columns in the dataset

# In[8]:


df.info()


# ### Check for null values

# In[9]:


df.isnull().sum()


# # Data exploration

# 1. Numerical columns=2
# 2. Categorical columns=4

# ### Unique values in each column

# ### 1. Age

# In[11]:


print("Number of Unique elements in Age column: ", df["Age"].nunique())
print("Maximum Age: ", max(df["Age"]))
print("Minimum Age: ", min(df["Age"]))
print()
print("Unique elements : \n", df["Age"].unique())


# ### 2. Sex

# In[12]:


print("Unique Elements: ", df["Sex"].unique())
print("Number of Unique elements in Sex column: ", df["Sex"].nunique())


# ### 3. BP

# In[14]:


print("Unique Elements: ", df["BP"].unique())
print("Number of Unique elements in BP column: ", df["BP"].nunique())


# ### 4. Cholestrol

# In[15]:


print("Unique elements : ", df["Cholesterol"].unique())
print("Number of Unique elements in cholesterol column: ", df["Cholesterol"].nunique())


# ### 5. Na_to_K

# In[16]:


print("Number of Unique elements in Na_to_K: ", df["Na_to_K"].nunique())
print()
print("Unique elements : \n", df["Na_to_K"].unique())


# ### 6. Drug

# In[17]:


print("Unique elements :", df["Drug"].unique())
print("Number of Unique elements in Drug column: ", df["Drug"].nunique())


# In[18]:


skewAge = df.Age.skew(axis = 0, skipna = True)
print('Age skewness: ', skewAge)


# In[20]:


sns.displot(df["Age"] , kde=True, bins = 20)
plt.show()


# In[21]:


skewNatoK = df.Na_to_K.skew(axis = 0, skipna = True)
print('Na to K skewness: ', skewNatoK)


# In[22]:


sns.displot(df["Na_to_K"] , kde=True, bins = 20)
plt.show()


# # Data Visualization and EDA

# ### 1. Distribution of males and females

# In[23]:


df.Sex.value_counts()


# In[24]:


sns.countplot(x='Sex', data = df, palette="Blues").set_title('Distribution of males and females')
plt.savefig('Distribution of males and females.png', transparent=True)


# ### 2. Distribution of different BP level

# In[25]:


df.BP.value_counts()


# In[26]:


sns.countplot(x='BP', data = df, palette="icefire").set_title('Distribution of different BP level')
plt.savefig('Distribution of different BP level.png', transparent=True)


# ### 3. Distribution of different cholesterol levels

# In[27]:


df.Cholesterol.value_counts()


# In[28]:


sns.countplot(x='Cholesterol', data = df, palette="cubehelix").set_title('Distribution of different cholesterol levels')
plt.savefig('Distribution of different cholesterol levels.png', transparent=True)


# ### 4. Distribution of different drugs

# In[29]:


df.Drug.value_counts()


# In[30]:


sns.countplot(x='Drug', data = df).set_title('Distribution of different drugs')
plt.savefig('Distribution of different drugs.png', transparent=True)


# ### 5. Gender distribution based on drug type

# In[31]:


pd.crosstab(df.Sex,df.Drug).plot(kind="barh",figsize=(12,5),color=['#00085c','#41a0a3','#58508d','#bc5090','#4d005c'])
plt.title('Gender distribution based on Drug type')
plt.xlabel('Frequency')
plt.xticks(rotation=0)
plt.ylabel('Gender')
plt.show()


# ### 6. Frequency - Sex vs Cholesterol

# In[32]:


sns.barplot(x = "Sex", y = "Count", hue = "Cholesterol", data = df.groupby(["Sex", "Cholesterol"]).size().reset_index(name = "Count"), palette="flare").set(title = "Frequency - Sex vs Cholesterol")


# ### 7. Frequency - Sex vs BP

# In[33]:


sns.barplot(x = "Sex", y = "Count", hue = "BP", data = df.groupby(["Sex", "BP"]).size().reset_index(name = "Count"), palette="Blues").set(title = "Frequency - Sex vs BP")


# ### 8. BP based on Cholesterol

# In[34]:


df.groupby(["BP","Cholesterol"])["Drug"].count().reset_index()


# In[35]:


pd.crosstab(df.BP,df.Cholesterol).plot(kind="barh",figsize=(15,6),color=['#28135c','#c90832'])
plt.title('Blood Pressure distribution based on Cholesterol')
plt.xlabel('Frequency')
plt.xticks(rotation=0)
plt.ylabel('BP')
plt.show()


# ### 9. Na_to_K based on gender and age

# In[36]:


plt.scatter(x=df.Age[df.Sex=='F'], y=df.Na_to_K[(df.Sex=='F')], c="red")
plt.scatter(x=df.Age[df.Sex=='M'], y=df.Na_to_K[(df.Sex=='M')], c="black")
plt.legend(["Female", "Male"])
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.show()


# ### 10. Drugs prescribed to sex

# In[37]:


plt.figure(figsize = (9,5))
sns.swarmplot(x = "Sex", y = "Drug",data = df)
plt.legend(df.Drug.value_counts().index)
plt.title("Sex Vs Drug")
plt.show()


# 1. Drug Y is prescribed the most to males and females
# 2. Drug B is prescribed the least to females
# 3. Drug C is prescribed the least to males

# In[38]:


df_Sex_Drug = df.groupby(["Drug","Sex"]).size().reset_index(name = "Count")
df_Sex_Drug


# ### 11. Drugs prescribed according to BP level

# In[39]:


df.plot(kind='scatter',x='BP',y='Drug', title = "BP vs drugs", color = "black", s=100)
plt.savefig('Cholesterol vs drugs.png', transparent=True)


# In[40]:


df_BP_Drug = df.groupby(["Drug","BP"]).size().reset_index(name = "Count")
df_BP_Drug


# 1. People having High BP are prescribed only drug Y,A,B
# 2. People having Low BP are prescibed only drug Y,C,X
# 3. People having Normal BP are prescribed drug X,Y

# ### 12. Drugs prescribed according to Cholesterol level

# In[42]:


df.plot(kind='scatter',x='Cholesterol',y='Drug', title = "Cholesterol vs drugs", color = "blue", s=100)
plt.savefig('Cholesterol vs drugs.png', transparent=True)


# 1. People with High cholesterol are described all drugs
# 2. People with normal cholesterol are described drug X,Y,A,B

# ### 13. Drugs prescribed according to Na_to_K

# In[43]:


df.plot(kind='scatter',x='Drug',y='Na_to_K', title = "Drug vs Na_to_K", color = "blue")
plt.savefig('Drug vs Na_to_K.png', transparent=True)


# 1. People with more than 15.015 Na_to_K are prescribed DrugY
# 2. People with less than 15.015 Na_to_K are prescribed DrugC, DrugX, DrugA, DrugB 

# ### 14. Drugs prescribed according to Age

# In[44]:


df.plot(kind='scatter',x='Drug',y='Age', title = "Drug vs Age", color = "red")
plt.savefig('Drug vs Age.png', transparent=True)


# People less than 50 age are not prescribed drug B and people greater than 51 age are not prescribed drug A

# In[45]:


print("Minimum Age of DrugB",df.Age[df.Drug == "drugB"].min())
print("Maximum Age of DrugA",df.Age[df.Drug == "drugA"].max())


# ### 15. Na_to_K Vs BP Vs Drug

# In[46]:


plt.figure(figsize = (9,5))
sns.swarmplot(x = "Drug", y = "Na_to_K",hue="BP",data = df, palette="flare")
plt.legend()
plt.title("Na_to_K Vs BP Vs Drug")
plt.show()


# # Data Modeling

# The age will be divided into 7 age categories:
# Age 
# 1. Below 20 y.o.
# 2. 20 - 29 y.o.
# 3. 30 - 39 y.o.
# 4. 40 - 49 y.o.
# 5. 50 - 59 y.o.
# 6. 60 - 69 y.o.
# 7. Above 70.

# In[47]:


bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df['Age_binned'] = pd.cut(df['Age'], bins=bin_age, labels=category_age)
df = df.drop(['Age'], axis = 1)


# In[48]:


binary_NatoK = [0, 9, 19, 29, 50]
categories_NatoK = ['<10', '10-20', '20-30', '>30']
df['Na_to_K_binned'] = pd.cut(df['Na_to_K'], bins=binary_NatoK, labels=categories_NatoK)
df = df.drop(['Na_to_K'], axis = 1)


# ### Splitting the dataset into train and test parts

# In[49]:


X = df.drop(["Drug"], axis=1)
y = df["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ### Feature engineering

# In[50]:


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# In[51]:


X_train.head()


# In[52]:


X_test.head()


# In[54]:


sns.set_theme(style="darkgrid")
sns.countplot(y=y_train, data=df, palette="coolwarm")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()


# # Models

# ### 1. Decision Tree

# In[58]:


DT = DecisionTreeClassifier(max_leaf_nodes=20)
DT.fit(X_train, y_train)

y_pred1 = DT.predict(X_test)

print("Classification Report: \n", classification_report(y_test, y_pred1))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred1))

Acc1 = accuracy_score(y_pred1,y_test)
print('\nDecision Tree accuracy is: {:.2f}%'.format(Acc1*100))


# In[59]:


scoreListDT = []
for i in range(2,50):
    DT = DecisionTreeClassifier(max_leaf_nodes=i)
    DT.fit(X_train, y_train)
    scoreListDT.append(DT.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListDT)
plt.xticks(np.arange(2,50,5))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.title("Leaf vs Score")
plt.show()
DTAccMax = max(scoreListDT)
print("DT Acc Max {:.2f}%".format(DTAccMax*100))


# ### Logistic Regression

# In[68]:


LR = LogisticRegression(solver='liblinear', max_iter=5000)
LR.fit(X_train, y_train)

y_pred3 = LR.predict(X_test)

print("Classification Report: \n", classification_report(y_test, y_pred3))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred3))

Acc2 = accuracy_score(y_pred3,y_test)
print('\nLogistic Regression accuracy is: {:.2f}%'.format(Acc3*100))


# ### Random Forest

# In[69]:


rf = RandomForestClassifier(max_leaf_nodes=30)
rf.fit(X_train, y_train)

y_pred4 = rf.predict(X_test)

print("Classification Report: \n", classification_report(y_test, y_pred4))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred4))

Acc3 = accuracy_score(y_pred4,y_test)
print('\nRandom Forest accuracy is: {:.2f}%'.format(Acc4*100))


# In[63]:


scoreListRF = []
for i in range(2,50):
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    rf.fit(X_train, y_train)
    scoreListRF.append(rf.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListRF)
plt.xticks(np.arange(2,50,5))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.title("RF value vs Score")
plt.show()
RFAccMax = max(scoreListRF)
print("RF Acc Max {:.2f}%".format(RFAccMax*100))


# ### K neighbours

# In[70]:


KN = KNeighborsClassifier(n_neighbors=20)
KN.fit(X_train, y_train)

y_pred_6 = KN.predict(X_test)

print("Classification Report: \n", classification_report(y_test, y_pred_6))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_6))

Acc_4 = accuracy_score(y_pred_6,y_test)
print('\nK Neighbours accuracy is: {:.2f}%'.format(Acc_6*100))


# In[72]:


scoreListknn = []
for i in range(1,30):
    KN = KNeighborsClassifier(n_neighbors = i)
    KN.fit(X_train, y_train)
    scoreListknn.append(KN.score(X_test, y_test))
    
plt.plot(range(1,30), scoreListknn)
plt.xticks(np.arange(1,30,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.title("K value vs Score")
plt.show()
KNAccMax = max(scoreListknn)
print("K Neighbours Acc Max {:.2f}%".format(KNAccMax*100))


# In[73]:


comparision = pd.DataFrame({'Model': ['Decision Tree', 'Decision Tree Max', 'Logistic Regression', 'Random Forest', 'Random Forest Max', 'K Neighbors', 'K Neighbors Max'], 
                        'Accuracy': [Acc1*100, DTAccMax*100, Acc2*100, Acc3*100, Acc4*100, RFAccMax*100, KNAccMax*100]})
comparision.sort_values(by='Accuracy', ascending=False)


# In[ ]:




