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
import clustering
#from skbio.core.distance import DissimilarityMatrix





if __name__ == '__main__':
    
    train_classes = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(0,1))
    train_shape_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(1,10))
    train_rgb_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(10,20))
    
    #dm1 = pdist(train_shape_view.as_matrix(), 'euclidean')
    #dm2 = pdist(train_rgb_view.as_matrix(), 'euclidean')

    hdCluster = clustering.hardClustering.HardClustering(7, 3, train_shape_view, train_rgb_view)
    for i in range(100):
    	if(hdCluster.run()):
    		print('Reached a local minimum')

    	print(str(i) + '/100 iterations done')

    hdCluster.printLog()

    knownCluster = clustering.gtClustering.GroundTruthClustering(train_classes)
    knownCluster.printLog()

    rand = clustering.randIndex.RandIndex(len(train_classes), hdCluster.getClusters(), knownCluster.getClusters())
    rand.printContingency()
    print("\n -- AJUSTED RAND INDEX -- \n", rand.getAdjusted())