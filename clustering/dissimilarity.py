import numpy as np
from scipy.spatial.distance import pdist
import random

class Dissimilarity(object):
    def __init__(self, view, weight=1):
        self.weight = weight
        self.matrix = np.arange(len(view)*len(view)).reshape(len(view), len(view))
        self.matrix.fill(0)
        tempView = np.asarray(view.values)

        for i in range(len(view)):
            for j in range(i, len(view)):
                mat = [tempView[i], tempView[j]]
                self.matrix[i][j] = pdist(mat, 'euclidean')
                #self.matrix[i][j] = np.linalg.norm(tempView[i]-tempView[j])
        return

    def get(self, i, j):
        if(j < i):
            i, j = j, i
        #print ('get: ' + str(i) + ', ' + str(j))
        #print ('dim: ' + str(len(self.matrix)) + ', ' + str(len(self.matrix[i])))
        #print ('weight: ' + str(self.weight) + ', mat[i][j]: ' + str(self.matrix[i][j]))
        
        return self.weight * self.matrix[i][j]

    def getUnweight(self, i, j):
        if(j < i):
            i, j = j, i

        #print ('get: ' + str(i) + ', ' + str(j))
        #print ('dim: ' + str(len(self.matrix)) + ', ' + str(len(self.matrix[i])))
        #print ('weight: ' + str(self.weight) + ', mat[i][j]: ' + str(self.matrix[i][j]))

        return self.matrix[i][j]        