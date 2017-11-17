import numpy as np
import clustering

class HardClustering(object):

    def getClosestCluster(self, point):
        cluster = -1
        closestDist = 999999
        for c in range(len(self.clusters)):
            curDist = self.clusters[c].prototype.dist(point, self.dissimilarities)
            if(curDist < closestDist):
                closestDist = curDist
                cluster = c

        return cluster


    def findBestPrototypes(self):
        
        for k in range(len(self.clusters)):
            #this should be ordered in ascending order by the first column
            closest = np.arange(self.n*2).reshape(self.n, 2)

            for i in range(self.n):
                d = self.clusters[k].dist(i, self.dissimilarities)
                closest[i][0] = d
                closest[i][1] = i

            #print('bef ', closest)
            closest = closest[closest[:,0].argsort()]
            #print('aft ', closest)
            prototypes = np.arange(self.clusters[k].prototype.getQ())
            for i in range(self.clusters[k].prototype.getQ()):
                prototypes[i] = closest[i][1]

            #print('bef ', self.clusters[k].prototype.prototypes)
            self.clusters[k].prototype.set(prototypes)
            #print('aft ', self.clusters[k].prototype.prototypes)

    def findBestWeights(self):
        p = len(self.dissimilarities)
        k = len(self.clusters)
        denominators = np.arange(p)
        numerator = 1.0
        for h in range(p):
            soma = 0
            for kzinho in range(k):
                soma += self.clusters[kzinho].distToPrototypeUnweight(self.dissimilarities[h])

            numerator *= soma
            denominators[h] = soma

        numerator = pow(numerator, 1.0/p)

        for j in range(p):
            self.dissimilarities[j].weight = numerator/denominators[j]

    def findBestPartition(self):
        done = True
        for point in range(self.n):
            shouldBelong = self.getClosestCluster(point)
            belongs = self.belongsTo[point]
            if(shouldBelong != belongs):
                done = False
                self.belongsTo[point] = shouldBelong
                self.clusters[belongs].remove(point)
                self.clusters[shouldBelong].insert(point)

        return done

    '''
    K = number of clusters
    q = number of repesentative elements
    view = dataset
    ''' 
    def __init__(self, k, q, n, shape, rgb):
        self.k = k
        self.q = q
        self.shape = shape
        self.rgb = rgb
        self.n = n
        self.t = 0
        #self.clusters = [None]*k
        self.clusters = []

        for i in range(k):
            #self.clusters[i] = clustering.cluster.Cluster(self.n, q)
            self.clusters.append(clustering.cluster.Cluster(self.n, q))

        self.dissimilarities = []
        self.dissimilarities.append(self.shape)
        self.dissimilarities.append(self.rgb)

        #self.belongsTo = [None]*self.n
        self.belongsTo = []
        for point in range(self.n):
            cluster = self.getClosestCluster(point)
            #self.belongsTo[point] = cluster
            self.belongsTo.append(cluster)
            self.clusters[cluster].insert(point)

        return 

    def findBestPartitions(self):
        self.t += 1
        self.findBestPrototypes()
        self.findBestWeights()
        done = self.findBestPartition()

        return done


    def getClusters(self):
        return self.clusters

    def printLog(self):
        print(self.getLog())

    def getLog(self):
        logStr = "\nDynamic Hard Clustering Parameters:\n"
        logStr += "T: " + str(self.t) + ", K: " + str(len(self.clusters)) + ", P: " + str(len(self.dissimilarities)) + ", Q: " + str(self.clusters[0].prototype.getQ()) +"\n"
        logStr += "\nPartitionedClusters:\n"
        logStr += "--------------------\n"
            
        for i in range(len(self.clusters)):
            logStr += "Cluster[" + str(i) + "]: " + self.clusters[i].tostr() + '\n\n'
        
        
        logStr += "\n\nPrototypes:\n"
        logStr += "-----------\n\n"

        for i in range(len(self.clusters)):
            logStr += "P[" + str(i) + "]: " + self.clusters[i].prototype.tostr() + '\n\n'
        
        logStr += "\nDissimilarity Matrix Weights:\n"
        logStr += "----------------------------\n\n"
        for i in range(len(self.dissimilarities)):
            logStr += "W[" + str(i) + "]:" + str(self.dissimilarities[i].weight) + '\n'

        logStr += '\n'
        return logStr
        

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
        print(self.getLog())
        # print("-- KNOWN CLUSTERING EXTRACTED FROM CLASS --\n");
        # for key in self.classMap:
        #     print(key + ': ', self.clusters[self.classMap[key]].tostr())
        #     print("")

    def getLog(self):
        logStr = "Ground Truth Clusters:\n"
        logStr += "----------------------\n"
        for key in self.classMap:
            logStr += key + ': ' + self.clusters[self.classMap[key]].tostr() + '\n\n'

        logStr += '\n'
        return logStr
