#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


# In[3]:


train=pd.read_csv('Data.csv')


# In[4]:


train.head()


# In[5]:


train


# In[6]:


#Lets upload the test data
test=pd.read_csv('Data_Test.csv')


# In[7]:


test.head()


# In[8]:


#lets check the shape of train and test data
train.shape


# In[9]:


test.shape


# In[10]:


#Information of train and test data
train.info()


# In[11]:


test.info()


# In[12]:


#lets describe the data of train and test set
train.describe


# In[13]:


test.describe


# In[14]:


#Lets check the null values on train and test set 
train.isnull().sum()


# There is no null value in train data set

# In[15]:


#Lets check on the test set
test.isnull().sum()


# In[16]:


#Lets use heatmap 
sns.heatmap(train.isnull())


# In[18]:


#Lets use correlation coefficient
train_cor=train.corr
train_cor


# In[19]:


#lets use heatmap on correlation
sns.heatmap(train.corr())


# In[20]:


#lets check the skewness
train.skew


# In[21]:


#Lets use countplot 
col=['malignant','highly_malignant','loathe','rude','abuse','threat']
for i in col:
    print(i)
    print("\n")
    print(train[i].value_counts())
    sns.countplot(train[i])
    plt.show()


# # Natural LanguageProcessing

# In[22]:


from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import  stopwords
import string


# In[23]:


train['length'] = train['comment_text'].str.len()
train.head()


# In[24]:


# Convert all messages to lower case
train['comment_text'] = train['comment_text'].str.lower()


# In[25]:


# Replace email addresses with 'email'
train['comment_text'] = train['comment_text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')


# In[26]:


# Replace URLs with 'webaddress'
train['comment_text'] = train['comment_text'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')


# In[27]:


# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
train['comment_text'] = train['comment_text'].str.replace(r'£|\$', 'dollers')


# In[28]:


# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
train['comment_text'] = train['comment_text'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')


# In[29]:


train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join(
    term for term in x.split() if term not in string.punctuation))


# In[30]:


stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])
train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))


# In[31]:


import nltk
nltk.download('wordnet')
lem=WordNetLemmatizer()
train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join(
lem.lemmatize(t) for t in x.split()))


# In[33]:


train['clean_length'] = train.comment_text.str.len()
train.head()


# In[34]:


# Total length removal
print ('Origian Length', train.length.sum())
print ('Clean Length', train.clean_length.sum())


# In[35]:


cols_target = ['malignant','highly_malignant','rude','threat','abuse','loathe']
df_distribution = train[cols_target].sum()                            .to_frame()                            .rename(columns={0: 'count'})                            .sort_values('count')


# In[36]:


df_distribution.plot.pie(y='count',title='Label distribution over comments',
                                      figsize=(5, 5))\
                            .legend(loc='center left', bbox_to_anchor=(1.3, 0.5))


# In[37]:


target_data = train[cols_target]

train['bad'] =train[cols_target].sum(axis =1)
print(train['bad'].value_counts())
train['bad'] = train['bad'] > 0 
train['bad'] = train['bad'].astype(int)
print(train['bad'].value_counts())


# In[38]:


sns.set()
sns.countplot(x="bad" , data = train)
plt.show()


# In[39]:


#  Convert text into vectors using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec = TfidfVectorizer(max_features = 10000, stop_words='english')
features = tf_vec.fit_transform(train['comment_text'])
x = features


# In[40]:


train.shape


# In[41]:


test.shape


# In[42]:


y=train['bad']
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=56,test_size=.30)


# In[43]:


y_train.shape,y_test.shape


# # Model Evaluation

# # Logistic Regression

# In[44]:


# LogisticRegression
from sklearn.linear_model import LogisticRegression
LG = LogisticRegression(C=1, max_iter = 3000)
LG.fit(X_train, y_train)


# In[45]:


y_pred_train = LG.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, y_pred_train)))
y_pred_test = LG.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))


# # Decision Tree

# In[46]:


# DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)


# In[47]:


y_pred_train = DT.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, y_pred_train)))
y_pred_test = DT.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))


# # Random Forest classifier

# In[49]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)


# In[50]:


y_pred_train = RF.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, y_pred_train)))
y_pred_test = RF.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))


# In[51]:


#KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)


# In[52]:


y_pred_train = knn.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, y_pred_train)))
y_pred_test = knn.predict(X_test)
print('Test accuracy is {}'.format(accuracy_score(y_test,y_pred_test)))
print(confusion_matrix(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))


# In[ ]:




