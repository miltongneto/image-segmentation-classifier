import numpy as np
import random

class Prototype(object):
    def __init__(self, q):
        self.q = q
        self.prototypes = []
        return

    def getQ(self):
        return self.q

    def randomize(self, n):
        self.prototypes = []
        for i in range(self.q):
            self.prototypes.append(int(random.uniform(0, n)))

    def set(self, prototypes):
        self.prototypes = prototypes

    def dist(self, j, dissimilarities):
        d = 0

        for i in range(self.q):
            for k in range(len(dissimilarities)):
                d += dissimilarities[k].get(self.prototypes[i], j)

        return d

    def distUnweight(self, j, dissimilarity):
        d = 0
        for i in range(self.q):
            d += dissimilarity.getUnweight(self.prototypes[i], j)
        
        return d
    
    def tostr(self):
        mystr = ''
        counter = 0
        for i in range(len(self.prototypes)):
            print()
            counter += 1

            if counter == 10:
                mystr += '\n'
                counter = 0

            mystr += str(self.prototypes[i]) + ", "

        return mystr
