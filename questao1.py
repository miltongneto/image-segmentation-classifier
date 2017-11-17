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

if __name__ == '__main__':
    iterations = 1000
    train_classes = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(0,1))
    train_shape_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(1,10))
    train_rgb_view = pd.read_csv('train.csv', header=0, index_col=None, usecols=range(10,20))
    
    dic = {}
    bestRandIndex = -99999
    for i in range(iterations):
        hdCluster = clustering.hardClustering.HardClustering(7, 3, train_shape_view, train_rgb_view)
        find = hdCluster.findMinimum()
        while not find:
            find = hdCluster.findMinimum()

        log = hdCluster.getLog()
        knownCluster = clustering.gtClustering.GroundTruthClustering(train_classes)
        log += knownCluster.getLog()

        rand = clustering.randIndex.RandIndex(len(train_classes), hdCluster.getClusters(), knownCluster.getClusters())
        log += rand.getLog()
        randIndex = rand.getAdjusted()
        dic[randIndex] = log
        if randIndex > bestRandIndex:
            bestRandIndex = randIndex

        print("Actual rand index: ", randIndex)    
        print(i+1, '/', iterations,' iterations done')

    print(dic[bestRandIndex])
    print("Best rand index: ", bestRandIndex)

    f = open('q1_output.txt','w')
    f.write(dic[bestRandIndex])
    f.write("\nBest rand index: " + str(bestRandIndex))
    f.close()
