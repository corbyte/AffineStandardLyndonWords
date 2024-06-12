import numpy as np
import argparse
from scipy import sparse
from sympy.liealgebras.root_system import RootSystem as sympy_RootSystem
class letter:
    index:int
    value:str
    rootIndex:int
    def __init__(self, v:str="",rootIndex:int=0):
        self.value = v
        self.rootIndex = rootIndex
    def set_index(self, i:int):
        self.index = i
    def __str__(self):
        return self.value
    def __lt__(self, other):
        return self.index < other.index
    def __eq__(self,other):
        if not isinstance(other,self.__class__):
            return NotImplemented
        return self.index == other.index
    def __ne__(self,other):
        return not (self.index == other.index)
    def __geq__(self, other):
        return not (self < other)
    def __hash__(self):
        return hash(self.rootIndex)
    def __le__(self, other):
        return self.index <= other.index
    def __ge__(self,other):
        return self.index >= other.index
class word:
    #Maybe change to sparse matrix
    def __init__(self, wordArray,l,imaginary=False,es=None,hs=None):
        self.string = np.array(wordArray,dtype=letter)
        self.imaginary=imaginary
        self.hs = None
        self.weights = np.zeros(l,dtype=int)
        for i in self.string:
            self.weights[i.rootIndex-1] += 1
        self.weights.flags.writeable = False
    def __len__(self):
        return len(self.string)
    def __getitem__(self,i):
        return self.string[i]
    def __str__(self):
        return ','.join(str(i) for i in self.string)
    def __eq__(self,other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if(len(self.string) != len(other.string)):
            return False
        for i in range(len(self.string)):
            if(self.string[i] != other.string[i]):
                return False
        return True
    def __lt__(self,other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        for i in range(min(len(self.string),len(other.string))):
            if(self.string[i] < other.string[i]):
                return True
            if(self.string[i] > other.string[i]):
                return False
        if(len(self.string) < len(other.string)):
            return True
        return False
    def __le__(self,other):
        return (self < other) or (self == other)
    def __gt__(self, other):
        return not (self <= other)
    def __ge__(self,other):
        return not (self < other)
    def __hash__(self):
        return hash(self.string.tobytes())
    def __ne__(self,other):
        return not (self == other)
    def __add__(self,other):
        return word(np.concatenate((self.string,other.string),dtype=word),len(self.weights)) 
class letterOrdering:
    def __init__(self, letterOrdering):
        letterOrdering = [letter(str(i),i) for i in letterOrdering]
        self.order:list[letter] = letterOrdering
        for i in range(len(letterOrdering)):
            self.order[i].index = i
    def __len__(self):
        return len(self.order)
    def __getitem__(self,index):
        return self.order[index]
    def __str__(self):
        return '<'.join([str(i) for i in self.order])

class rootSystem:
    def eBracket(self, A,B):
        if(self.isImaginary(A.weights)):
            re = B
            im = A
        else:
            re = A
            im = B
        weights = re.weights - (self.delta * sum(re.weights) // sum(self.delta))
        return np.any(np.logical_and(weights!= 0, (im.hs @self.cartan_matrix) != 0))
    def hBracket(self,A,B):
        #FIXME:
        newA = (A.weights - (self.delta *A.weights[-1]))
        newB = (B.weights - (self.delta * B.weights[-1]))
        if(np.any(newA < 0)):
            return newB
        return newA
    def __init__(self, ordering,type,k=0):
        self.arr = []
        type = type.upper()
        self.k = k
        self.affine = k != 0
        if( len(type) != 1 or type < 'A' or type > 'G' ):
            raise ValueError('Type is invalid')
        if(self.affine):
            self.n = len(ordering)-1
        else:
            self.n = len(ordering)
        self.ordering:letterOrdering = letterOrdering(ordering)
        self.arr.append([word([i],len(self.ordering)) for i in self.ordering.order])
        self.weightToWordDictionary = {}
        for i in self.arr[0]:
            self.weightToWordDictionary[i.weights.tobytes()] = [i]
        if(type == 'A'):
            multiDimBase = rootSystem.getAWeights(self.n,self.affine)
        elif(type == 'C'):
            multiDimBase = rootSystem.getCWeights(self.n,self.affine)
        elif(type == 'G'):
            multiDimBase = rootSystem.getGWeights(self.affine)
        self.baseWeights = np.array([i for j in multiDimBase for i in j],dtype=object)
        if(self.affine):
            self.cartan_matrix = np.zeros((self.n+1,self.n+1), dtype=int)
            self.cartan_matrix[:-1,:-1] = np.array(sympy_RootSystem(type +str(self.n)).cartan_matrix(),dtype=int)
            if(type == 'A'):
                self.delta = rootSystem.TypeADelta(self.n)
                extension = np.zeros(self.n+1,dtype=int)
                extension[0] = -1
                extension[-2] = -1
                extension[-1] = 2
                self.cartan_matrix[:,-1] = extension
                self.cartan_matrix[-1] = extension
            elif (type == 'B'):
                self.delta = rootSystem.TypeBDelta(self.n) 
                extension[1] = -1
                extension[-1] = 2
                self.cartan_matrix[:-1] = extension
                self.cartan_matrix[:,:-1] = extension
            elif(type == 'C'):
                self.delta = rootSystem.TypeCDelta(self.n)
                self.cartan_matrix[-1,-1]= 2
                self.cartan_matrix[-1,0] = -1
                self.cartan_matrix[0,-1] = -2
            elif(type == 'G'):
                self.delta = rootSystem.TypeGDelta()
                self.cartan_matrix[-1,-1] = 2
                self.cartan_matrix[-1,0] = -1
                self.cartan_matrix[0,-1] = -1
            self.deltaWeight = sum(self.delta)
            #Generates the words
            self.__genAffine()
        else:
            if(type == 'A'):
                self.__genFinite()
            elif(type == 'B'):
                self.__genTypeBFinite()
            elif(type == 'C'):
                self.__genFinite()
            elif(type == 'D'):
                self.__genTypeDFinite()
            elif(type == 'G'):
                self.__genFinite()
    def __genFinite(self):
        for i in self.baseWeights:
            self.__genWord(i)
    def __genAffine(self):
        for k in range(self.k+1):
            for i in self.baseWeights:
                self.__genWord(i + self.delta * k)
    def __genTypeBFinite(self):
        size = self.n
        for length in range(2,2*size):
            #i
            if(length <= size):
                comb = np.zeros(size,dtype=int)
                for i in range(size-length,size):
                    comb[i] = 1
                self.__genWord(comb)
                if(length != size):
                    #ei - ej
                    for start in range(0,size - length):
                        comb = np.zeros(size,dtype=int)
                        for k in range(start,start+length):
                            comb[k] = 1
                        self.__genWord(comb)
            #ei + ej
            if(length >= 3):
                comb = np.zeros(size,dtype=int)
                comb[-1] = 2
                for i in range(size-min(length-1,size),size-1):
                    comb[i] = 1
                oneIndex = size-min(length-1,size)
                twoIndex = size-1
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex] = 2
                while(oneIndex < twoIndex):
                    self.__genWord(comb)
                    twoIndex -= 1
                    comb[twoIndex] = 2
                    comb[oneIndex] = 0
                    oneIndex+=1    
    def __genTypeDFinite(self):
        size=self.n
        for length in range(2,2*size-2):
            #i-j
            for start in range(0,size - length):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k] = 1
                self.__genWord(comb)
            #i+j
            if(length < size):
                comb=np.zeros(size,dtype=int)
                comb[-1] =1
                for i in range(-(length+1),-2,1):
                    comb[i] = 1
                self.__genWord(comb)
            if(length >= 3):
                comb=np.zeros(size,dtype=int)
                comb[-1] = 1
                comb[-2] = 1
                for i in range(size-min(length,size),size-2):
                    comb[i] = 1
                oneIndex = size-min(length,size)
                twoIndex = size-2
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex] = 2
                while(oneIndex < twoIndex):
                    self.__genWord(comb)
                    twoIndex -= 1
                    comb[twoIndex] = 2
                    comb[oneIndex] = 0
                    oneIndex+=1
    def getAWeights(n,affine=False):
        size = n
        if(affine):
            size += 1
        arr = [[] for i in range(size)]
        for length in range(1,n+1):
            for start in range(0,n - length + 1):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k] = 1
                arr[length-1].append(comb)
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeADelta(n))
        return arr    
    def __genAffineBaseWeights(arr,delta):
        for length in range(1,len(arr)):
            for i in arr[-length-1]:
                if(i is None or i[-1] == 1):
                    break
                arr[length-1].append(delta - i)
            arr[-1] = np.array([None])
            arr[-1][0] = delta
    def getGWeights(affine:bool=False):
        if(affine):
            arr = [[] for i in range(6)]
        else:
            arr = [[] for i in range(5)]
        for i in [[1,0],[0,1],[1,1],[1,2],[1,3],[2,3]]:
            if(affine):
                i.append(0)
            arr[sum(i)-1].append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeGDelta())
        return arr
    def getCWeights(n,affine:bool=False):
        size = n
        if(affine):
            size+=1
            arr = [[] for i in range(2*size-1)]
        else:
            arr = [[] for i in range(1,2*n)]
        for i in range(n):
            comb = np.zeros(size,dtype=int)
            comb[i] = 1
            arr[0].append(comb)
        for length in range(2,2*n):
            if(length <= 2*n -2):
                #i+j
                comb = np.zeros(size,dtype=int)
                i = n-1
                while i >= 0 and n-i <= length:
                    comb[i] = 1
                    i -= 1
                oneIndex = i+1
                twoIndex = n-1
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex] = 2
                while(oneIndex < twoIndex):
                    arr[length-1].append(np.copy(comb))
                    twoIndex -= 1
                    comb[twoIndex] = 2
                    comb[oneIndex] = 0
                    oneIndex+=1
            #i-j
            if(length < n):
                for start in range(0,n-length):
                    comb = np.zeros(size,dtype=int)
                    for k in range(start,start+length):
                        comb[k] =1
                    arr[length-1].append(comb)
            #2i
            if(length % 2 == 1):
                comb=np.zeros(size,dtype=int)
                comb[n-1] = 1
                for k in range(n-2,n-2-length//2,-1):
                    comb[k] = 2
                arr[length-1].append(comb)
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeCDelta(n))
        return arr
    def standardFactorization(self,wordToFactor):
        if(type(wordToFactor) is not word):
            res = self.getWords(np.array(wordToFactor,dtype=int))
            if(len(res) == 0):
                raise ValueError('Combination not found')
            if(len(res) != 1):
                raise ValueError('Must give real word')
            wordToFactor = res[0]
        wordToFactorWeight = wordToFactor.weights
        weight = np.copy(wordToFactor.weights)
        for i in range(len(wordToFactor)):
            weight[wordToFactor[i].rootIndex -1] -= 1
            rightWords = self.getWords(weight)
            if(len(rightWords) == 0):
                continue
            for rightWord in rightWords:
                flag = False
                for j in range(len(rightWord)):
                    if(rightWord[j] != wordToFactor[i+j+1]):
                        flag = True
                        break
                if(flag):
                    continue
                return (self.getWords(wordToFactorWeight-weight)[0],rightWord)
    def getWords(self, combination):
        combination = np.array(combination,dtype=int)
        if(not combination.tobytes() in self.weightToWordDictionary):
            return []
        else:
            return self.weightToWordDictionary[combination.tobytes()]
    def getAffineWords(self,weight):
        if not self.affine:
            raise ValueError('Cannot call getAffineWords on a simple Lie algebra')
        matches = []
        k=0
        newWord = self.getWords(weight + k*self.delta)
        while len(newWord) > 0:
            matches.extend(newWord)
            k+=1
            newWord=self.getWords(weight + k*self.delta)
        return matches
    def isImaginary(self,combinations):
        return (self.affine and sum(combinations) % self.deltaWeight == 0)
    def __genWord(self, combinations):
        if(type(combinations)is not np.array):   
            combinations = np.array(combinations,dtype=int)
        weight = sum(combinations)
        if(weight == 1):
            return
        imaginary = self.isImaginary(combinations)
        potentialOptions = []
        if (weight > len(self.arr)+1):
            return None
        minSubRoot  = self.arr[0][0]
        for i in self.arr[0]:
            if i < minSubRoot:
                minSubRoot = i
        for i in self.baseWeights:
            i = np.copy(i)
            j = combinations-i
            if(len(self.getWords(j)) == 0):
                continue
            iImaginary = self.isImaginary(i)
            jImaginary = self.isImaginary(j)
            while(min(j) >= 0 and sum(i) <= len(self.arr)):
                if(self.affine and iImaginary and jImaginary):
                    break
                words1 = self.getWords(i)
                for word1 in words1:
                    if(word1 < minSubRoot and not imaginary):
                        continue
                    words2 = self.getWords(j)
                    for word2 in words2:
                        if(word2 < minSubRoot and not imaginary):
                            continue
                        if(word1< word2):
                            (a,b) = (word1,word2)
                        else:
                            (a,b) = (word2,word1)
                        newWord = a + b
                        if(self.affine and (iImaginary or jImaginary)):
                            bracket = self.eBracket(a,b)
                            #Checks to see if bracket is non-zero
                            if not bracket.any():
                                continue
                        minSubRoot = a
                        potentialOptions.append((newWord,a,b))
                if(not self.affine):
                    break
                i += self.delta
                j -= self.delta
        if(len(self.arr) < weight):
            self.arr.append([])
        if not imaginary:
            match = potentialOptions[-1][0]
            self.arr[weight-1].append(match)
            self.weightToWordDictionary[combinations.tobytes()] = [match]
        else:
            potentialOptions = list(set(potentialOptions))
            potentialOptions.sort(reverse=True)
            matrix = np.zeros((self.n+1,self.n+1), dtype = int)
            while(not self.hBracket(potentialOptions[0][1],potentialOptions[0][2]).any()):
                potentialOptions.pop(0)
            potentialOptions[0][0].hs = self.hBracket(potentialOptions[0][1],potentialOptions[0][2])
            liPotentialOptions = [potentialOptions[0][0]]
            matrix[0] = potentialOptions[0][0].hs
            index = 1
            row = 1
            while(row < self.n):
                #change to only use non-zero matrix entries
                potentialOptions[index][0].hs = self.hBracket(potentialOptions[index][1],potentialOptions[index][2])
                matrix[row] = potentialOptions[index][0].hs
                if(np.linalg.matrix_rank(matrix) == row+1):
                    liPotentialOptions.append(potentialOptions[index][0])
                    row+=1
                index += 1
            self.arr[-1].extend(liPotentialOptions)
            self.weightToWordDictionary[combinations.tobytes()] = liPotentialOptions
    def getBaseWeights(self):
        return np.array(self.baseWeights)
    def getWordsByBase(self):
        returnarr = []
        for i in self.getBaseWeights():
            returnarr.append(np.array(self.getAffineWords(i)))
        return np.array(returnarr)
    def getMonotonicity(self, comb):
        weights = [i.weights for i in self.getAffineWords(comb)]
        for j in range(1,len(weights)):
            monotonicity = 0
            if(self.getWords(weights[j])[0] < self.getWords(weights[j-1])[0]):
                if(monotonicity == -1):
                    monotonicity = 0
                    break
                monotonicity = 1
            else:
                if(monotonicity == 1):
                    monotonicity = 0
                    break
                monotonicity = -1
        return monotonicity
    def checkMonotonicity(self, filter:{'All', 'Increasing', 'Decreasing','None'}="All"):
        returnarr = []
        for i in self.getBaseWeights()[:-1]:
            monotonicity = self.getMonotonicity(i)
            if(filter == 'None' and monotonicity != 0):
                continue
            if(filter == 'Increasing' and monotonicity != 1):
                continue
            if(filter == 'Decreasing' and monotonicity != -1):
                continue
            returnarr.append((self.getWords(i)[0].weights,monotonicity))
        return np.array(returnarr, dtype=object)
    def checkConvexity(self):
        exceptions = []
        for length in range(len(self.arr)):
            for sumWord in self.arr[length]:
                for j in range(length):
                    for alphaWord in self.arr[j]:
                        diff = sumWord.weights - alphaWord.weights
                        if(min(diff) < 0):
                            continue
                        for betaWord in self.getWords(diff):
                            if( betaWord<sumWord == alphaWord < sumWord):
                                exceptions.append((betaWord,alphaWord))
        return exceptions
    def TypeADelta(n:int):
        return np.ones(n+1,dtype=int)
    def TypeBDelta(n:int):
        #TODO:
        return [0]
    def TypeCDelta(n:int):
        delta = np.repeat(2,n+1)
        delta[-1] = 1
        delta[-2] = 1
        return delta
    def TypeDDelta(n:int):
        #TODO:
        return[0]
    def TypeGDelta():
        return np.array([2,3,1],dtype=int)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type",choices=["C","c","A","a","b","B",'d','D'])
    parser.add_argument("size",type=int)
    parser.add_argument("-o","--order", nargs='+', type =int)
    parser.add_argument('-a','--affine_count',type=int, default=0)
    args = parser.parse_args()
    #args = parser.parse_args(['C','6', '-a' ,'6'])
    type = args.type
    affineCount = args.affine_count
    size = args.size
    orderInput = args.order
    # parameter input: type affinecount size stanard order y/n order if n
    order = []
    if(orderInput is None):
        if(affineCount == 0):
            order = [int(i) for i in range(1,size+1)]
        else:
            type = type.upper()
            if(type == 'A'):
                order = [int(i) for i in range(1,size+1)]
                order.append(0)
            elif(type =='C'):
                order = [i for i in range(-1,size)]
                order[0] = size
            else:
                order = [int(i) for i in range(1,size+1)]
                order.append(0)
    else:
        order = [int(i) for i in orderInput]
    rootsystem = rootSystem(order,type,affineCount) 
if __name__ == '__main__':    
    main()