# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:52:08 2022

@author: Asus
"""

import pandas as pd;
import numpy as np;
import nltk;
from nltk.corpus import stopwords;
import re;
from nltk.stem import WordNetLemmatizer;
from sklearn.feature_extraction.text import CountVectorizer;

nltk.download('omw-1.4');

dataset = pd.read_csv("Restaurant_Reviews.tsv", 
                      sep='\t', quoting=3);

x = dataset.iloc[:,0].values;
y = dataset.iloc[:,-1].values;

stop_words = stopwords.words('english');
stop_words.remove('not');

wnl = WordNetLemmatizer();
cv = CountVectorizer();
independent_vars = [];
ans = [];


for row in dataset.to_dict('records'):
    rev = row['Review'];
    ans.append(row['Liked']);
    rev = re.sub('[^a-zA-Z]',' ', rev);
    rev = rev.split();
    temp = [];    
    for word in rev:
        if word not in stop_words:
            temp.append(wnl.lemmatize(word.lower()));
    independent_vars.append(' '.join(temp));

independent_vars = np.array(independent_vars);
independent_vars = cv.fit_transform(independent_vars);
ans = np.array(ans);


from sklearn.model_selection import train_test_split;
x_train, x_test, y_train, y_test = train_test_split(independent_vars, ans,
                                        test_size = 0.2, random_state = 32);


from sklearn.naive_bayes import BernoulliNB;
model = BernoulliNB();
model.fit(x_train, y_train);
y_test_pred = model.predict(x_test);



from sklearn.metrics import accuracy_score;
score = accuracy_score(y_test, y_test_pred);
print(score)


from sklearn.svm import SVC;
model = SVC();
model.fit(x_train, y_train);
y_test_pred = model.predict(x_test);

score = accuracy_score(y_test, y_test_pred);
print(score)

from sklearn.ensemble import RandomForestClassifier;
model = RandomForestClassifier(n_estimators = 40);
model.fit(x_train, y_train);
y_test_pred = model.predict(x_test);

score = accuracy_score(y_test, y_test_pred);
print(score)