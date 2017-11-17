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
    dataPath = 'train.csv'

    print('Starting Hard Clustering Algorithm')
    iterations = 100
    print('Reading classes')
    train_classes = pd.read_csv(dataPath, header=0, index_col=None, usecols=range(0,1))
    print('Reading shape view')
    train_shape_view = pd.read_csv(dataPath, header=0, index_col=None, usecols=range(1,10))
    print('Reading rgb view')
    train_rgb_view = pd.read_csv(dataPath, header=0, index_col=None, usecols=range(10,20))
    print('End reading')

    print('Creating shape dsim matrix')
    dsimShape = clustering.dissimilarity.Dissimilarity(train_shape_view)
    print('Creating rgb dsim matrix')
    dsimRgb = clustering.dissimilarity.Dissimilarity(train_rgb_view)

    dic = {}
    bestRandIndex = -99999
    for i in range(iterations):
        hdCluster = clustering.algorithms.HardClustering(7, 3, len(train_classes), dsimShape, dsimRgb)
        find = False

        maxIterationsUntilFindPartitions = 10000
        it = 1
        while not find or it > maxIterationsUntilFindPartitions:
            find = hdCluster.findBestPartitions()
            it += 1

        log = hdCluster.getLog()
        knownCluster = clustering.algorithms.GroundTruthClustering(train_classes)
        log += knownCluster.getLog()

        rand = clustering.randIndex.RandIndex(len(train_classes), hdCluster.getClusters(), knownCluster.getClusters())
        log += rand.getLog()
        randIndex = rand.getAdjusted()
        dic[randIndex] = log
        if randIndex > bestRandIndex:
            bestRandIndex = randIndex

        print("Adjusted Rand Index: ", randIndex)    
        print('Iteration ', i+1, '/', iterations)

    print(dic[bestRandIndex])
    print("Adjusted Rand Index: ", bestRandIndex)

    f = open('q1_output.txt','w')
    f.write(dic[bestRandIndex])
    f.write("\nAdjusted Rand Index: " + str(bestRandIndex))
    f.close()
