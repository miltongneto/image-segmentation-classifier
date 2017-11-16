import numpy as np
import clustering

class HardClustering(object):

    def closestCluster(self, point):
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
            #this should be ordered in descending order by the first column
            closest = np.arange(self.n*2).reshape(self.n, 2)

            for i in range(self.n):
                d = self.clusters[k].dist(i, self.dissimilarities)
                closest[i][0] = d
                closest[i][1] = i

            closest = closest[closest[:,0].argsort()]

            prototypes = np.arange(self.clusters[k].prototype.getQ())
            for i in range(self.clusters[k].prototype.getQ()):
                prototypes[i] = closest[0][1]
                closest = np.delete(closest, 0, 0)

            self.clusters[k].prototype.set(prototypes)

    def findBestWeights(self):
        p = len(self.dissimilarities)
        k = len(self.clusters)
        denominators = np.arange(p)
        numerator = 1
        for h in range(p):
            soma = 0
            for kzinho in range(k):
                soma += self.clusters[kzinho].distToPrototypeUnweight(self.dissimilarities[h])

            numerator *= soma
            denominators[h] = soma

        numerator = pow(numerator, 1.0/p)

        for j in range(p):
            self.dissimilarities[j].weigth = numerator/denominators[j]


    def defineBestPartition(self):
        stuck = True
        for point in range(self.n):
            shouldBelong = self.closestCluster(point)
            belongs = self.belongsTo[point]
            if(shouldBelong != belongs):
                stuck = False
                self.belongsTo[point] = shouldBelong
                self.clusters[belongs].remove(point)
                self.clusters[shouldBelong].insert(point)

        return stuck

    '''
    K = number of clusters
    q = no idea
    view = dataset
    ''' 
    def __init__(self, k, q, shape, rgb):
        self.k = k
        self.q = q
        self.shape = shape
        self.rgb = rgb
        self.n = len(shape)
        self.t = 0
        self.clusters = [None]*k

        for i in range(k):
            self.clusters[i] = clustering.cluster.Cluster(self.n, q)

        self.dissimilarities = [None]*2
        self.dissimilarities[0] = clustering.dissimilarity.Dissimilarity(self.shape)
        self.dissimilarities[1] = clustering.dissimilarity.Dissimilarity(self.rgb)

        self.belongsTo = [None]*self.n
        for point in range(self.n):
            cluster = self.closestCluster(point)
            self.belongsTo[point] = cluster
            self.clusters[cluster].insert(point)

        return 

    def run(self):
        self.t += 1
        self.findBestPrototypes()
        self.findBestWeights()
        stuck = self.defineBestPartition()

        return stuck


    def getClusters(self):
        return self.clusters

    def printLog(self):
        print("\n\n[Hard-Clustering] NEW ITERATION:\n")
        print("T: " + str(self.t) + ", K: " + str(len(self.clusters)) + ", P: " + str(len(self.dissimilarities)) + ", Q: " + str(self.clusters[0].prototype.getQ()) +"\n");
        print("\n -- CLUSTER STATE --\n");
            
        for i in range(len(self.clusters)):
            print("[" + str(i) + "]: ", self.clusters[i].tostr())
            print("")
        
        print("\n -- PROTOTYPE STATE --\n");

        for i in range(len(self.clusters)):
            print("[" + str(i) + "]: ", self.clusters[i].prototype.tostr())
            print("")
        
        print("\n -- WEIGHT STATE --\n");
        for i in range(len(self.dissimilarities)):
            print("[" + str(i) + "]:" + str(self.dissimilarities[i].weight))
        