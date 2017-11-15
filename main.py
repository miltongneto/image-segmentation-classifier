import numpy as np
from scipy import stats
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier


class Classifier(object):

  def __init__(self):
    self.train_shape_view = None
    self.train_rgb_view = None
    return

  def readViews(self):
      self.train_shape_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(1,10))
      self.train_rgb_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(10,20))
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
      clf = GaussianNB()
      clf.fit(train_set_predictors, train_set_labels)
      return clf

  def knn(self, train_set_predictors, train_set_labels):
      k = self.findK(train_set_predictors, train_set_labels)
      clf = KNeighborsClassifier(n_neighbors=k, p=1)
      clf.fit(train_set_predictors, train_set_labels)
      return clf

  def testClassifier(self, clf, test_set_predictors, test_set_labels):
      target_pred = clf.predict(test_set_predictors)
      test_accuracy = accuracy_score(test_set_labels, target_pred)
      print(test_accuracy)

      return

  def findK(self, train_set_predictors, train_set_labels):
    cv_scores = []
    ks = [1, 3, 5, 7, 9, 11, 13]
    for k in ks:
      knn = KNeighborsClassifier(n_neighbors=k, p=1)
      scores = cross_val_score(knn, train_set_predictors, train_set_labels, cv=10, scoring='accuracy')
      mean = scores.mean()
      cv_scores.append(mean)

    mse = [1 - x for x in cv_scores]
    optimal_k = ks[mse.index(min(mse))]

    return optimal_k

  def classifierCombination(self, train_set_predictors_v1, train_set_labels_v1, test_set_predictors_v1, train_set_predictors_v2, train_set_labels_v2, test_set_predictors_v2, test_set_labels):
    #TODO: PARALELIZAR O TREINAMENTO
    bayes_view1 = self.bayes(train_set_predictors_v1, train_set_labels_v1)
    bayes_view2 = self.bayes(train_set_predictors_v2, train_set_labels_v2)
    knn_view1 = self.knn(train_set_predictors_v1, train_set_labels_v1)
    knn_view2 = self.knn(train_set_predictors_v2, train_set_labels_v2)

    # print(test_set_labels.size)
    targets_pred = np.empty(shape=(0,test_set_labels.size))

    targets_pred = np.vstack((targets_pred, bayes_view1.predict(test_set_predictors_v1)))
    targets_pred = np.vstack((targets_pred, bayes_view2.predict(test_set_predictors_v2)))
    targets_pred = np.vstack((targets_pred, knn_view1.predict(test_set_predictors_v1)))
    targets_pred = np.vstack((targets_pred, knn_view2.predict(test_set_predictors_v2)))

    target_pred = stats.mode(targets_pred).mode[0]
    test_accuracy = accuracy_score(test_set_labels, target_pred)
    print(test_accuracy)
    return

  def process(self, view):
    self.readViews()
    if view == "shape":
      print("Shape View:")
      train_set_predictors, train_set_labels, test_set_predictors, test_set_labels = classifier.preProcess(classifier.train_shape_view)
    elif view == "rgb":
      print("RGB View:")
      train_set_predictors, train_set_labels, test_set_predictors, test_set_labels = classifier.preProcess(classifier.train_rgb_view)
    else:
      train_set_predictors_shape, train_set_labels_shape, test_set_predictors_shape, test_set_labels = classifier.preProcess(classifier.train_shape_view)
      train_set_predictors_rgb, train_set_labels_rgb, test_set_predictors_rgb, test_set_labels = classifier.preProcess(classifier.train_rgb_view)
      self.classifierCombination(train_set_predictors_shape, train_set_labels_shape, test_set_predictors_shape, train_set_predictors_rgb, train_set_labels_rgb, test_set_predictors_rgb, test_set_labels)
      return

    print("-Bayes:")
    clf = classifier.bayes(train_set_predictors, train_set_labels)
    classifier.testClassifier(clf, test_set_predictors, test_set_labels)
    print("-KNN:")
    clf = classifier.knn(train_set_predictors, train_set_labels)
    classifier.testClassifier(clf, test_set_predictors, test_set_labels)

    return

def majorityVote(self, train_set_predictors, train_set_labels, test_set_predictors, target_pred):
  c_bayes = GaussianNB()
  k = self.findK(train_set_predictors, train_set_labels)
  c_knn = KNeighborsClassifier(n_neighbors=k, p=1)
  classifier_final = VotingClassifier(estimators=[('bayes', c_bayes), ('knn', c_knn)], voting='hard')
  classifier_final.fit(train_set_predictors, train_set_labels)

  target_pred = classifier_final.predict(test_set_predictors)
  test_accuracy = accuracy_score(test_set_labels, target_pred)
  print(test_accuracy)

  return

#classifier = Classifier()
#classifier.process(None)
# classifier.process("shape")
# classifier.process("rgb")
