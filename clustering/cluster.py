import numpy as np
import clustering

class Cluster(object):
    def __init__(self, n, q):
        self.n = n
        self.q = q
        self.prototype = clustering.prototype.Prototype(self.q)
        self.prototype.randomize(self.n)
        self.elements = []

        return

    def insert(self, value):
        self.elements.append(value)

    def remove(self, value):
        self.elements.remove(value)

    def dist(self, j, dissimilarities):
        d = 0
        for element in self.elements:
            for k in range(len(dissimilarities)):
                d+= dissimilarities[k].get(element, j)

        return d

    def distToPrototypeUnweight(self, dissimilarity):
        d = 0;
        for element in self.elements:
            d += self.prototype.distUnweight(element, dissimilarity)
            
        return d;
        
    def printStuff(self):
        for element in self.elements:
            print(str(element) + ", ")
