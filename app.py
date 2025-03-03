#!/usr/bin/env python
# coding: utf-8

# **😰 Stress Detection from Social Media Articles**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import express

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score, confusion_matrix,f1_score

from keras.models import Sequential


# In[2]:


REDDIT = 'Twitter_Full.csv'
USECOLS = ['text','hashtags', 'labels']

df = pd.read_csv(filepath_or_buffer=REDDIT, sep=';', usecols=USECOLS)
df.head()
df.shape


# In[3]:


df.sample(10)


# In[3]:


df['labels'].value_counts()


# In[ ]:


#REDDIT = 'Reddit_Combi.csv'
#USECOLS = ['Body_Title', 'label']

#df = pd.read_csv(filepath_or_buffer=REDDIT, sep=';', usecols=USECOLS)
#df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df['labels'].value_counts()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df = df.drop_duplicates()
df


# In[11]:


df = df.dropna()
df


# In[12]:


""""""plt.pie(df['labels'].value_counts(),labels=df['labels'].unique())
#plt.legend()
#plt.show()""""""


# In[13]:


df['text'].value_counts()


# In[18]:


X = df['text']
y = df['labels']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


tfidf_vectorizer = TfidfVectorizer()


# In[21]:


X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[27]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print('List of stop words:', stop_words)


# In[28]:


pip install nltk


# In[29]:


import nltk
print(nltk.data.path)


# In[32]:


import nltk
from nltk.corpus import stopwords


# In[31]:


nltk.download("stopwords")


# In[33]:


def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        df[column][index] = string


# In[34]:


nlp_preprocessing


# In[35]:


import os


# In[36]:


from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[37]:


get_ipython().system('pip install wordcloud')


# In[40]:


from wordcloud import WordCloud


# In[41]:


combined_text_pos = ' '.join(df[df['labels']==1])
combined_text_neg = ' '.join(df[df['labels']==0])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(combined_text_pos)
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(combined_text_neg)


# In[ ]:





# In[42]:


df.sample(10)


# Removing punctuations

# In[43]:


import string


# In[44]:


exclude = string.punctuation


# In[45]:


def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))


# In[46]:


text = df['text']


# In[47]:


print(df['text'])


# In[48]:


df


# In[49]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=df[df['labels']==1]['text'].str.len()
ax1.hist(tweet_len,color='blue')
ax1.set_title('stressfree text')

tweet_len=df[df['labels']==0]['text'].str.len()
ax2.hist(tweet_len,color='red')
ax2.set_title('Stressful text')
fig.suptitle('Characters in tweets')
plt.show()


# In[50]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len = df[df['labels']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='blue')
ax1.set_title('Stressful text')

tweet_len=df[df['labels']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='red')
ax2.set_title('stressfree text')
fig.suptitle('Words in a tweet')
plt.show()


# In[51]:


df = df.drop('hashtags',axis=1)


# In[52]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
nltk.download('stopwords')


# In[53]:


print("Max index in DataFrame:", df.index.max())
print("DataFrame length:", len(df))


# In[54]:


corpus = []

for index, row in df.iterrows():
    # Extract text and perform operations
    text = re.sub('[^A-Za-z\s]', '', str(row['text']))
    text = text.lower()
    corpus.append(text)


# In[55]:


corpus


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


cv.fit_transform(corpus).toarray()


# In[ ]:


X = cv.fit_transform(corpus).toarray()


# In[ ]:


y = df['labels']


# In[ ]:


print("Shape of X:", X.shape)
print("Shape of Y:", y.shape)


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2, random_state=11)


# # Model Training

# **1: Logistic Regression**

# In[ ]:


clf = LogisticRegression()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[ ]:


print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **2: Random forest classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


RFC=RandomForestClassifier()


# In[ ]:


RFC.fit(X_train,y_train)


# In[ ]:


predRFC = RFC.predict(X_test)


# In[ ]:


accuracy_score(y_test,predRFC)


# In[ ]:


print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **3: Decision tree classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier()


# In[ ]:


dt.fit(X_train, y_train)


# In[ ]:


preddt = dt.predict(X_test)


# In[ ]:


accuracy_score(y_test,preddt)


# In[ ]:


print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **4: Support vector classifier**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc = SVC()


# In[ ]:


svc.fit(X_train, y_train)


# In[ ]:


pred_svc = svc.predict(X_test)


# In[ ]:


accuracy_score(y_test,pred_svc)


# In[ ]:


print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **5: MultinomialNB**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()
clf1.fit(X_train,y_train)


# In[ ]:


# Predicting the Test set results
y_pred = clf1.predict(X_test)


# In[ ]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)


# In[ ]:


print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **6: BernoulliNB**

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
clf2 = BernoulliNB()
clf2.fit(X_train,y_train)


# In[ ]:


# Predicting the Test set results
y_pred = clf2.predict(X_test)


# In[ ]:


print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# **7: Artificial Neural network**

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])


# In[ ]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)


# In[ ]:


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[ ]:





# In[ ]:


classifiers = {
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'Multinomial Naive Bayes': MultinomialNB()
}


# In[ ]:


# Train and evaluate each classifier
for name, clf in classifiers.items():
    print("Classifier:", name)
    clf.fit(X_train, y_train)
    
    # Predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    error = 1 - accuracy
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Error:", error)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("="*80)


# In[ ]:


#pip install yellowbrick


# In[ ]:


from yellowbrick.classifier import ClassificationReport


# In[ ]:


accuracy_scores = {}
f1_scores = {}
recall_scores = {}
precision_scores = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy_scores[name] = accuracy_score(y_test, y_pred)
    f1_scores[name] = f1_score(y_test, y_pred)
    recall_scores[name] = recall_score(y_test, y_pred)
    precision_scores[name] = precision_score(y_test, y_pred)


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 22))

for ax, (metric, scores) in zip(axes.flatten(), [('Accuracy', accuracy_scores), ('F1 Score', f1_scores), ('Recall', recall_scores), ('Precision', precision_scores)]):
    ax.bar(scores.keys(), scores.values(), color='skyblue')
    ax.set_title(metric)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)  # Setting y-axis limit between 0 and 1 for better visualization
    ax.grid(axis='y', linestyle='--', alpha=0.9)

plt.tight_layout()
plt.show()


# In[ ]:





# ## Testing our model manually 

# In[ ]:


a = 'very bad'
a_cv = cv.transform([a]).toarray()


# In[ ]:


clf.predict(a_cv)


# In[ ]:


b = 'good , i got internship'


# In[ ]:


b_cv = cv.transform([b]).toarray()


# In[ ]:


clf.predict(b_cv)


# In[ ]:


c = 'two girls died in college'


# In[ ]:


c_cv = cv.transform([c]).toarray()


# In[ ]:


clf.predict(c_cv)


# In[ ]:


df['text'][8643]


# In[ ]:


d = 'I was very good at class & so was Dad face with tears of joy so I got loads of sausage, been for a walk this afternoon did lots of sniffing and said hello to some overseas lorry drivers, off to the doggie Drs soon her name is Dr Biscuit face with tears of joy face with tears of joy dogs dogsoftwitter dogslife happy'


# In[ ]:


d_cv = cv.transform([d]).toarray()


# In[ ]:


clf.predict(d_cv)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


classifiers = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [None, 5, 10, 15]}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
    'Multinomial NB': (MultinomialNB(), {'alpha': [0.1, 1, 10]}),
    'Bernoulli NB': (BernoulliNB(), {'alpha': [0.1, 1, 10]}),
    'ANN': (MLPClassifier(max_iter=1000), {'hidden_layer_sizes': [(50,), (100,), (50, 50)]})
}

# Perform grid search for each classifier
for clf_name, (clf, param_grid) in classifiers.items():
    print(f"Grid search for {clf_name}...")
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X, y)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




