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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class Classifier(object):

  def __init__(self):
    self.train_shape_view = None
    self.train_rgb_view = None
    return

  def readViews(self):
      self.train_shape_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=[*range(0,10)])
      self.train_rgb_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=[0, *range(10,20)])
      self.test_shape_view = pd.read_csv('test.csv', header=0, index_col=None, usecols=[*range(0,10)])
      self.test_rgb_view = pd.read_csv('test.csv', header=0, index_col=None, usecols=[0, *range(10,20)])

      return

  # Some data for training and other part for test (90% and 10%)
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

  # All data for training or test
  def preProcessCompleteData(self, dataset_df):
    train_set_predictors = dataset_df.drop('CLASS', axis=1)
    train_set_labels = dataset_df['CLASS'].copy()

    le = preprocessing.LabelEncoder()
    le.fit(dataset_df['CLASS'])
    
    train_set_labels_numeric = le.transform(train_set_labels)

    return train_set_predictors, train_set_labels_numeric

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
    cv_scores2 = []
    scores2 = []
    ks = [1, 3, 5, 7, 9, 11, 13]
    
    for k in ks:
      knn = KNeighborsClassifier(n_neighbors=k, p=1)
      scores = cross_val_score(knn, train_set_predictors, train_set_labels, cv=10, scoring='accuracy')
      mean = scores.mean()
      cv_scores.append(mean)
    
    mse = [1 - x for x in cv_scores]
    optimal_k = ks[mse.index(min(mse))]

    return optimal_k

  def cross_validate(self, k, clf, train_set_predictors, train_set_labels):
    skf = StratifiedKFold(n_splits=k)
    scores = []
    predictors_set = np.array(train_set_predictors)
    label_set = np.array(train_set_labels)
        
    for train_index, test_index in skf.split(predictors_set, label_set):
      clf.fit(predictors_set[train_index], label_set[train_index])
      target_pred = clf.predict(predictors_set[test_index])
      score = accuracy_score(label_set[test_index], target_pred)
      scores.append(score)
      mean = np.mean(scores);
    return mean

  def combinationClassifier(self, train_set_predictors_v1, train_set_predictors_v2, train_set_labels):
    #TODO: PARALELIZAR O TREINAMENTO
    self.bayes_view1 = self.bayes(train_set_predictors_v1, train_set_labels)
    self.bayes_view2 = self.bayes(train_set_predictors_v2, train_set_labels)
    self.knn_view1 = self.knn(train_set_predictors_v1, train_set_labels)
    self.knn_view2 = self.knn(train_set_predictors_v2, train_set_labels)

    return

  def testFinalClassifier(self, test_set_predictors_v1, test_set_predictors_v2, test_set_labels):
    targets_pred = np.empty(shape=(0,test_set_labels.size))

    targets_pred = np.vstack((targets_pred, self.bayes_view1.predict(test_set_predictors_v1)))
    targets_pred = np.vstack((targets_pred, self.bayes_view2.predict(test_set_predictors_v2)))
    targets_pred = np.vstack((targets_pred, self.knn_view1.predict(test_set_predictors_v1)))
    targets_pred = np.vstack((targets_pred, self.knn_view2.predict(test_set_predictors_v2)))

    target_pred = stats.mode(targets_pred).mode[0]
    test_accuracy = accuracy_score(test_set_labels, target_pred)
    print("Final Classifier: ")
    print(test_accuracy)
    return

  def comparingModels(self, train_set_predictors_v1, train_set_predictors_v2, train_set_labels):
    targets_pred = np.empty(shape=(0,train_set_labels.size))

    print("Bayes:")
    print("Shape View")
    score1 = cross_val_score(GaussianNB(), train_set_predictors_v1, train_set_labels, cv=30)
    print(np.mean(score1))
    
    print("RGB View")
    score2 = cross_val_score(GaussianNB(), train_set_predictors_v2, train_set_labels, cv=30)
    print (np.mean(score2))

    print("k-NN:")
    print("Shape View")
    k = self.findK(train_set_predictors_v1, train_set_labels)
    clf = KNeighborsClassifier(n_neighbors=k, p=1)
    score3 = cross_val_score(clf, train_set_predictors_v1, train_set_labels, cv=30)
    print (np.mean(score3))

    print("RGB View")
    k = self.findK(train_set_predictors_v2, train_set_labels)
    clf = KNeighborsClassifier(n_neighbors=k, p=1)
    score4 = cross_val_score(clf, train_set_predictors_v2, train_set_labels, cv=30)
    print (np.mean(score4))

    return

  def process(self):
    self.readViews()

    train_set_predictors_shape, train_set_labels_shape = classifier.preProcessCompleteData(classifier.train_shape_view)
    train_set_predictors_rgb, train_set_labels_rgb = classifier.preProcessCompleteData(classifier.train_rgb_view)
    self.comparingModels(train_set_predictors_shape, train_set_predictors_rgb, train_set_labels_shape)
    
    self.combinationClassifier(train_set_predictors_shape, train_set_predictors_rgb, train_set_labels_shape)
    
    test_set_predictors_shape, test_set_labels_shape = classifier.preProcessCompleteData(classifier.test_shape_view)
    test_set_predictors_rgb, test_set_labels_rbg = classifier.preProcessCompleteData(classifier.test_rgb_view)
    self.testFinalClassifier(test_set_predictors_shape, test_set_predictors_rgb, test_set_labels_shape)
    return

  def stratifiedCrossvalidate(self):
    return

  def majorityVote(self, train_set_predictors, train_set_labels, test_set_predictors, target_pred):
    c_bayes = GaussianNB()
    k = self.findK(train_set_predictors, train_set_labels)
    c_knn = KNeighborsClassifier(n_neighbors=k, p=1)
    final_classifier = VotingClassifier(estimators=[('bayes', c_bayes), ('knn', c_knn)], voting='hard')
    final_classifier.fit(train_set_predictors, train_set_labels)

    target_pred = final_classifier.predict(test_set_predictors)
    test_accuracy = accuracy_score(test_set_labels, target_pred)
    print(test_accuracy)

    return

classifier = Classifier()
classifier.process()
