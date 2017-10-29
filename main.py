import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class Classifier(object):

  def __init__(self):
    self.train_shape_view = None
    self.train_rgb_view = None
    self.clf = None
    return

    
  def readViews(self):
      self.train_shape_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=[*range(0,10)])
      self.train_rgb_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=[0, *range(10,20)])
      #test_df = pd.read_csv('test.csv', header=0, index_col=None)
      
      return
  
  def preProcess(self, train_df):
      train_set, test_set = train_test_split(train_df, test_size= 0.1, random_state = 42)

      train_set_predictors = train_set.drop('CLASS', axis=1)
      train_set_labels = train_set['CLASS'].copy()
      test_set_predictors = test_set.drop('CLASS', axis=1)
      test_set_labels = test_set['CLASS'].copy()

      le = preprocessing.LabelEncoder()
      le.fit(train_set['CLASS'])
      
      train_set_labels_numeric = le.transform(train_set_labels)    
      test_set_labels_numeric = le.transform(test_set_labels)
      
      return train_set_predictors, train_set_labels_numeric, test_set_predictors, test_set_labels_numeric
  
  def bayes(self, train_set_predictors, train_set_labels):
      self.clf = GaussianNB()
      self.clf.fit(train_set_predictors, train_set_labels)
      return
  
  def knn(self, train_set_predictors, train_set_labels):
      k = self.findK(train_set_predictors, train_set_labels)
      self.clf = KNeighborsClassifier(n_neighbors=k, p=1)
      self.clf.fit(train_set_predictors, train_set_labels)
      return
  
  def testClassifier(self, test_set_predictors, test_set_labels):
      target_pred = self.clf.predict(test_set_predictors)
      
      test_accuracy = accuracy_score(test_set_labels, target_pred)
      print(test_accuracy)

      return 
  
  def findK(self, train_set_predictors, train_set_labels):
    cv_scores = []
    ks = [1, 3, 5, 7, 9, 11, 13]
    for k in ks:
      #print(k)
      knn = KNeighborsClassifier(n_neighbors=k, p=1)
      scores = cross_val_score(knn, train_set_predictors, train_set_labels, cv=10, scoring='accuracy')
      mean = scores.mean()
      #print(mean)
      cv_scores.append(mean)

    mse = [1 - x for x in cv_scores]
    optimal_k = ks[mse.index(min(mse))]
    #print(optimal_k)

    return optimal_k

  def process(self, view):
    self.readViews()
    if view == "shape":
      print("Shape View:")
      train_set_predictors, train_set_labels, test_set_predictors, test_set_labels = classifier.preProcess(classifier.train_shape_view)
    elif view == "rgb":
      print("RGB View:")
      train_set_predictors, train_set_labels, test_set_predictors, test_set_labels = classifier.preProcess(classifier.train_rgb_view)
    else:
      return
    
    print("-Bayes:")
    classifier.bayes(train_set_predictors, train_set_labels)
    classifier.testClassifier(test_set_predictors, test_set_labels)
    print("-KNN:")
    classifier.knn(train_set_predictors, train_set_labels)
    classifier.testClassifier(test_set_predictors, test_set_labels)
     
    return


classifier = Classifier()
classifier.process("shape")
classifier.process("rgb")

