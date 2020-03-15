
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt


# In[133]:


import os 
print(os.getcwd())


# In[134]:


file_path=r"C:\Users\anbo\Downloads\train (3) (1) (3) (2).csv"
train = pd.read_csv(file_path)


# In[135]:


train.head()


# In[136]:


train['Review Title'].fillna('',inplace=True)
train.head()


# In[137]:


features = train.iloc[:, 3:5].values
print(features)


# In[138]:


labels = train.iloc[:,5]
print(labels)


# In[139]:


processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


# In[140]:


processed_features


# In[141]:


import nltk
nltk.download('stopwords')


# In[142]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


# In[143]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# In[144]:


from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(processed_features, labels)


# In[145]:


test_path=r"C:\Users\anbo\Downloads\test (3) (1) (3) (2).csv"
test= pd.read_csv(test_path)


# In[146]:


test.head()


# In[147]:


test['Review Title'].fillna('',inplace=True)


# In[148]:


x_test = test.iloc[:,3:5].values
print(x_test)


# In[149]:


processed_test_features = []

for sentence in range(0, len(x_test)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_test_features.append(processed_feature)


# In[150]:


processed_test_features


# In[151]:


processed_test_features = vectorizer.transform(processed_test_features).toarray()


# In[152]:


predictions = text_classifier.predict(processed_test_features)


# In[153]:


columns = ['Star Rating']
ratings = pd.DataFrame(predictions.reshape(len(predictions)),columns=columns)


# In[154]:


ratings


# In[155]:


ids = test.iloc[:,0:1]


# In[156]:


ids


# In[157]:


result = pd.concat([ids,ratings],axis=1,sort=False)


# In[158]:


result


# In[159]:


result.to_csv('predictions.csv',encoding='utf-8', index=False)

