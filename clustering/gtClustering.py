import numpy as np
import clustering

class GroundTruthClustering(object):
    def __init__(self, classes):
        self.classMap = {}
        self.clusters = []
        self.c = 0

        for i in range(len(classes)):
            label = classes.values[i][0]
            
            if label not in self.classMap:
                self.classMap[label] = self.c
                self.c += 1
                self.clusters.append(clustering.cluster.Cluster(len(classes), 0))

            self.clusters[self.classMap[label]].insert(i)

        return

    def getClusters(self):
        return self.clusters
    
    def printLog(self):
        print("-- KNOWN CLUSTERING EXTRACTED FROM CLASS --\n");
        for key in self.classMap:
            print(key + ': ', self.clusters[self.classMap[key]].tostr())
            print("")
