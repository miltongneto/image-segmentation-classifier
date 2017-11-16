import numpy as np
import clustering

class RandIndex(object):
    def __init__(self, n, a, b):
        self.a = a
        self.b = b
        self.adjusted = 0
        self.n = n
        self.rSum = [0]*len(a)
        self.cSum = [0]*len(b)

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
        print("\n -- CONTINGENCY TABLE -- \n\n");
        for i in range(len(self.contingency)):
            mystr = ''
            for j in range(len(self.contingency[i])):
                mystr += str(self.contingency[i][j]) + ' '
            
            print(mystr, "| ", self.rSum[i]);
        
        mystr = ''
        for j in range(len(self.contingency[0])):
            mystr += "___ "
        print(mystr)

        mystr = ''
        print("\n");
        for j in range(len(self.contingency[0])):
            mystr += str(self.cSum[j]) + ' '
        
        print(mystr)
        
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
