import numpy as np
import clustering

class RandIndex(object):
    def __init__(self, n, a, b):
        self.a = a
        self.b = b
        self.adjusted = 0
        self.n = n
        self.rSum = np.arange(len(a))
        self.rSum.fill(0)
        self.cSum = np.arange(len(b))
        self.cSum.fill(0)

        self.contingency = np.arange(len(a)*len(b)).reshape(len(a), len(b))
        for i in range(len(a)):
            for j in range(len(b)):
                self.contingency[i][j] = self.intersection(a[i], b[j])
                self.rSum[i] += self.contingency[i][j]
                self.cSum[j] += self.contingency[i][j]
        self.calculateAdjustedIndex()

        return

    def getAdjusted(self):
        return self.adjusted

    def printContingency(self):
        print(self.getLog())

    def getLog(self):
        myStr = "\nClusters Intersection ([rows]HardCluster x GroundTruth[columns]):\n"
        myStr += "------------------\n\n"
        myStr += '  Ground Truth\n'
        for i in range(len(self.contingency)):
            if i == 0:
                myStr += 'H '
            elif i == 1:
                myStr += 'a '
            elif i == 2:
                myStr += 'r '
            elif i == 3:
                myStr += 'd '
            else:
                myStr += '  '
            for j in range(len(self.contingency[i])):
                val = str(self.contingency[i][j])
                for k in range(5-len(val)):
                    val += ' '
                myStr += val + ' '
            
            myStr += "| " + str(self.rSum[i]) + '\n'
        
        myStr += ' '
        for j in range(len(self.contingency[0])):
            myStr += "______"
        myStr += '\n'

        myStr += '\n'
        myStr += ' '
        for j in range(len(self.contingency[0])):
            val = ' ' + str(self.cSum[j])
            for k in range(5-len(val)):
                    val += ' '
            myStr += val + ' '
        
        myStr += '\n'
        return myStr
        
    def intersection(self, a, b):
        soma = 0
        for element in a.elements:
            if element in b.elements:
                soma += 1
        return soma

    def comb(self, n):
        if n < 2:
            return 0
        else:
            return (n*(n-1))/2

    def calculateAdjustedIndex(self):
        index = 0

        for i in range(len(self.contingency)):
            for j in range(len(self.contingency[i])):
                index += self.comb(self.contingency[i][j])

        expectedIndex = 0
        maxIndex = 0
        left = 0
        right = 0

        for element in self.rSum:
            left += self.comb(element)
            maxIndex += self.comb(element)

        for element in self.cSum:
            right += self.comb(element)
            maxIndex += self.comb(element)

        expectedIndex = (left*right)/self.comb(self.n)
        maxIndex /= 2
        self.adjusted = (index-expectedIndex)/(maxIndex-expectedIndex)
