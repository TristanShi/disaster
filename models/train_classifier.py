# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Tristan SHI

@File: train_classifier.py
@Time: 2020/4/22 14:56
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle  # pickle模块

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
def load_data(database_filepath: str)->(np.array, np.array, pd.core.indexes.base.Index):
    """
    load data from database
    :param database_filepath:
    :return:
     X: training set, text.
     Y: training label
     category_names: training label's columns'names
    """
    # load data from database
    engine = create_engine('sqlite:////home/workspace/data/DisasterMessages.db')
    df = pd.read_sql_table(database_filepath, engine)
    X = np.array(df['message'])
    Y = np.array(df[df.columns[5:]])
    category_names = df.columns[5:]
    return X, Y, category_names


def tokenize(text:str) ->list :
    """
    clean text data: lower, remove stop words, Lemmatization
    :param text: message text
    :return: tokens
    """
    stop_words = nltk.corpus.stopwords.words('english')
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = ([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
    return tokens


def build_model() ->sklearn.pipeline.Pipeline:
    """
    build pipline model
    :return: model
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = [
        {"clf": [RandomForestClassifier()],
         "clf__n_estimators": [10, 100, 250],
         "clf__max_depth":[8],
         "clf__random_state":[42]
        }]
    model = GridSearchCV(model, param_grid=parameters, return_train_score=True,scoring='f1_micro')   
    return model


def evaluate_model(model:sklearn.pipeline.Pipeline, X_test: np.array, Y_test: np.array, category_names:pd.core.indexes.base.Index):
    """
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col, classification_report(Y_pred[:, i], Y_test[:, i]))
    return True


def save_model(model:sklearn.pipeline.Pipeline, model_filepath:str):
    """
    :param model: 
    :param model_filepath: 
    :return: 
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
