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
from main import Classifier
from scipy.spatial.distance import pdist
#from skbio.core.distance import DissimilarityMatrix





if __name__ == '__main__':

    classifier = Classifier()
    classifier.readViews()
    teste = classifier.train_shape_view.as_matrix();
    teste2 = classifier.train_rgb_view.as_matrix()

    #matrizes de dissimilaridade

    dm1 = pdist(teste, 'euclidean')
    dm2 = pdist(teste2, 'euclidean')
