import pandas as pd
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

data = 'spam.csv' # csv file for training the model 

df= pd.read_csv(f'./src/{data}', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

# dumping pickle file of transformed data 
trans_data = 'tranform.pkl'
pickle.dump(cv, open(f'./models/{trans_data}', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#dumping model
clf_pkl= 'nlp_model.pkl'
pickle.dump(clf, open(f'./models/{clf_pkl}', 'wb'))