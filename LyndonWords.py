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
    def __gt__(self,other):
        return self.index > other.index
    def __ne__(self,other):
        return not (self.index == other.index)
    def __hash__(self):
        return hash(self.rootIndex)
    def __le__(self, other):
        return self.index <= other.index
    def __ge__(self,other):
        return self.index >= other.index
class word:
    #Maybe change to sparse matrix
    def __init__(self, wordArray,weights):
        self.string = np.array(wordArray,dtype=letter)
        self.hs = None
        self.weights = weights
        self.height = sum(self.weights)
        self.weights.flags.writeable = False
        self.cofactorizationSplit = None
    def __len__(self):
        return self.height
    def __getitem__(self,i):
        return self.string[i]
    def __str__(self):
        return ','.join(str(i) for i in self.string)
    def __eq__(self,other):
        if(len(self.string) != len(other.string)):
            return False
        for i in range(len(self.string)):
            if(not self.string[i] == other.string[i]):
                return False
        return True
    def __lt__(self,other):
        return word.letterListCmp(self.string,other.string) < 0
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
        return word(np.concatenate((self.string,other.string),dtype=word),self.weights + other.weights) 
    def letterListCmp(first:np.array,second:np.array):
        lFirst = len(first)
        lSecond = len(second)
        if(lFirst < lSecond):
            minLen = lFirst
        else:
            minLen = lSecond
        for i in range(minLen):
            if(first[i].index > second[i].index):
                return 1
            if(first[i].index < second[i].index):
                return -1
        if(lFirst < lSecond):
            return -1
        if(lFirst > lSecond):
            return 1
        return 0
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
    def __iter__(self):
        self.iterIndex = 0
        return self
    def __next__(self) -> word:
        if(self.iterIndex == len(self.order)):
            raise StopIteration
        temp = self.order[self.iterIndex]
        self.iterIndex+=1
        return temp

class rootSystem:
    def eBracket(self, bracketWord:word):            
        (A,B) = self.costandardFactorization(bracketWord)
        if(A is None):
            return False
        if(self.isImaginary(A.height)):
            re = B
            im = A
        elif(self.isImaginary(B.height)):
            re = A
            im = B
        else:
            return True
        weights = re.weights - (self.delta * re.weights[-1])
        return np.any(weights[:-1]* (im.hs @self.cartan_matrix))
    def hBracket(self,word:word):
        a = self.costandardFactorization(word)[0]
        if(a is None):
            return np.zeros(self.n,dtype=int)
        word.cofactorizationSplit = a.height
        newA = (a.weights - (self.delta *a.weights[-1]))
        if(np.any(newA < 0)):
            return -newA[:-1]
        return newA[:-1]
    def __init__(self, ordering,type:str,k:int=0):
        type = type.upper()
        self.k = k
        self.affine:bool = k != 0
        if( len(type) != 1 or type < 'A' or type > 'G' ):
            raise ValueError('Type is invalid')
        if(self.affine):
            self.n = len(ordering)-1
        else:
            self.n = len(ordering)
        self.ordering:letterOrdering = letterOrdering(ordering)
        self.weightToWordDictionary:dict = {}
        self.minWord:word = None
        def weightsGeneration(letter):
            arr = np.zeros(len(self.ordering),dtype=int)
            arr[letter.rootIndex-1] = 1
            return arr
        for i in [word([i],weightsGeneration(i)) for i in self.ordering.order]:
            if(self.minWord is None):
                self.minWord = i
            elif(self.minWord > i):
                self.minWord = i
            i.cofactorizationSplit = 0
            self.weightToWordDictionary[i.weights.tobytes()] = [i]
        if(type == 'A'):
            self.baseWeights = rootSystem.getAWeights(self.n,self.affine)
        elif(type == 'B'):
            self.baseWeights = rootSystem.getBWeights(self.n,self.affine)
        elif(type == 'C'):
            self.baseWeights = rootSystem.getCWeights(self.n,self.affine)
        elif(type =='D'):
            self.baseWeights = rootSystem.getDWeights(self.n,self.affine)
        elif(type == 'F'):
            self.baseWeights = rootSystem.getFWeights(self.affine)
        elif(type == 'G'):
            self.baseWeights = rootSystem.getGWeights(self.affine)
        self.numberOfBaseWeights = len(self.baseWeights)
        #TODO: Maybe it'd be faster to just generate the base weights and then sort them by length
        if(self.affine):
            self.cartan_matrix =np.array(sympy_RootSystem(type +str(self.n)).cartan_matrix(),dtype=int)
            if(type == 'A'):
                self.delta = rootSystem.TypeADelta(self.n)
            elif (type == 'B'):
                self.delta = rootSystem.TypeBDelta(self.n) 
            elif(type == 'C'):
                self.delta = rootSystem.TypeCDelta(self.n)
            elif(type == 'D'):
                self.delta = rootSystem.TypeDDelta(self.n)
            elif(type == 'F'):
                self.delta = rootSystem.TypeFDelta()
            elif(type == 'G'):
                self.delta = rootSystem.TypeGDelta()
            self.deltaHeight = sum(self.delta)
            #Generates the words
            self.__genAffineRootSystem()
        else:
            self.__genFiniteRootSystem()
    def __genFiniteRootSystem(self):
        for i in self.baseWeights:
            self.__genWord(i)
    def __genAffineRootSystem(self):
        currentWeights = np.array(self.baseWeights,dtype=int)
        for _ in range(self.k+1):
            for i in currentWeights:
                self.__genWord(i)
                i += self.delta
    def getDWeights(n,affine=False):
        size=n
        if(affine):
            size = n+1
        arr = []
        for i in range(n):
            comb = np.zeros(size,dtype=int)
            comb[i] = 1
            arr.append(comb)
        for length in range(2,2*n-2):
            #i-j
            for start in range(0,n - length):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k] = 1
                arr.append(comb)
            #i+j
            if(length < n):
                comb=np.zeros(size,dtype=int)
                comb[n-1] =1
                for i in range(n-1-length,n-2):
                    comb[i] = 1
                arr.append(np.copy(comb))
            if(length >= 3):
                comb=np.zeros(size,dtype=int)
                comb[n-1] = 1
                comb[n-2] = 1
                for i in range(n-min(length,n),n-2):
                    comb[i] = 1
                oneIndex = n-min(length,n)
                twoIndex = n-2
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex] = 2
                while(oneIndex < twoIndex):
                    arr.append(np.copy(comb))
                    twoIndex -= 1
                    comb[twoIndex] = 2
                    comb[oneIndex] = 0
                    oneIndex+=1
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeDDelta(n))
        arr.sort(key = sum)
        return arr
    def getAWeights(n,affine=False):
        size = n
        if(affine):
            size += 1
        arr = []
        for length in range(1,n+1):
            for start in range(0,n - length + 1):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k] = 1
                arr.append(comb)
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeADelta(n))
        arr.sort(key = sum)
        return np.array(arr) 
    def getBWeights(n,affine=False):
        if(affine):
            size = n+1
        else:
            size = n
        arr = []
        for i in range(n):
            comb = np.zeros(size,dtype=int)
            comb[i] = 1
            arr.append(comb)
        for length in range(2,2*n):
            #i
            if(length <= n):
                comb = np.zeros(size,dtype=int)
                for i in range(n-length,n):
                    comb[i] = 1
                arr.append(comb)
                if(length != n):
                    #ei - ej
                    for start in range(0,n - length):
                        comb = np.zeros(size,dtype=int)
                        for k in range(start,start+length):
                            comb[k] = 1
                        arr.append(comb)
            #ei + ej
            if(length >= 3):
                comb = np.zeros(size,dtype=int)
                comb[n-1] = 2
                for i in range(n-min(length-1,n),n-1):
                    comb[i] = 1
                oneIndex = n-min(length-1,n)
                twoIndex = n-1
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex] = 2
                while(oneIndex < twoIndex):
                    arr.append(np.array(comb,dtype=int))
                    twoIndex -= 1
                    comb[twoIndex] = 2
                    comb[oneIndex] = 0
                    oneIndex+=1
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeBDelta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def __genAffineBaseWeights(arr,delta:np.array):
        for i in arr:
            if(i[-1] == 1):
                break
            arr.append(delta - i)
        arr.append(delta)
    def getGWeights(affine:bool=False)-> np.array:
        arr = []
        for i in [[1,0],[0,1],[1,1],[1,2],[1,3],[2,3]]:
            if(affine):
                i.append(0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeGDelta())
        arr.sort(key = sum)
        return np.array(arr)
    def getFWeights(affine:bool=False):
        arr = []
        for i in [[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,1],[0,1,1,1],
                  [0,1,2,2],[0,1,2,1],[1,1,1,1],[1,1,2,1],[1,2,2,1],[1,2,3,1],
                  [0,1,2,0],[1,1,2,0],[1,2,2,0],[1,2,2,2],[1,1,1,0],[0,1,1,0],
                  [1,3,4,2],[2,3,4,2],[1,1,2,2],[1,2,3,2],[1,1,0,0],[1,2,4,2]]:
            if(affine):
                i.append(0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeFDelta())
        arr.sort(key = sum)
        return np.array(arr)
    def getCWeights(n,affine:bool=False):
        size = n
        if(affine):
            size+=1
        arr = []
        for i in range(n):
            comb = np.zeros(size,dtype=int)
            comb[i] = 1
            arr.append(comb)
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
                    arr.append(np.copy(comb))
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
                    arr.append(comb)
            #2i
            if(length % 2 == 1):
                comb=np.zeros(size,dtype=int)
                comb[n-1] = 1
                for k in range(n-2,n-2-length//2,-1):
                    comb[k] = 2
                arr.append(comb)
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeCDelta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def costandardFactorization(self,wordToFactor:word):
        if(wordToFactor.height == 1):
            return (wordToFactor,None)
        weight = np.copy(wordToFactor.weights)
        weight[wordToFactor.string[0].rootIndex -1] -= 1
        splitLetter = None
        for i in self.ordering:
            if(weight[i.rootIndex-1] != 0):
                splitLetter = i
                break
        weight[wordToFactor.string[0].rootIndex -1] += 1
        for i in range(1,wordToFactor.height):
            weight[wordToFactor.string[i-1].rootIndex -1] -= 1
            if(wordToFactor.string[i].index != splitLetter.index):
                continue
            rightWords = self.getWords(weight)
            rightWord = None
            for rWord in rightWords:
                flag = True
                for j in range(sum(weight)):
                    if(rWord.string[j].index != wordToFactor.string[i+j].index):
                        flag=False
                        break
                if(flag):
                    rightWord = rWord
                    break
            if(rightWord is None):
                continue
            leftWord = None
            leftWords = self.getWords(wordToFactor.weights-weight)    
            for lWord in leftWords:
                flag = True
                for j in range(lWord.height):
                    if(lWord.string[j].index != wordToFactor.string[j].index):
                        flag=False
                        break
                if(flag):
                    leftWord = lWord
                    break
            if(leftWord is None):
                continue
            return (leftWord,rightWord)
        return (None,None)
    def getWords(self, combination:np.array):
        bytes = combination.tobytes()
        if(not bytes in self.weightToWordDictionary):
            return []
        else:
            return self.weightToWordDictionary[bytes]
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
    def isImaginary(self,height):
        return (self.affine and height % self.deltaHeight == 0)
    def __genWord(self, combinations:np.array):
        if(np.all(combinations == np.array([2,2,2,2,1]))):
            pass
        weight = sum(combinations)
        if(weight == 1):
            return
        imaginary = self.isImaginary(weight)
        potentialOptions = []
        maxWord  = self.minWord
        validBase = np.repeat(True,self.numberOfBaseWeights)
        kDelta = np.zeros(len(self.ordering),dtype=int)
        lengthChecked = weight
        i = self.baseWeights[0]
        iSum=0
        while(iSum <= lengthChecked):
            for baseWordIndex in range(self.numberOfBaseWeights):
                if(not validBase[baseWordIndex]):
                    continue
                i = self.baseWeights[baseWordIndex] + kDelta
                iSum = sum(i)
                if(iSum > lengthChecked):
                    break
                j = combinations-i
                words2 = self.getWords(j)
                if(len(words2) == 0):
                    validBase[baseWordIndex] = False
                    continue
                lengthChecked = weight - iSum
                eitherRootImaginary = (self.isImaginary(iSum) or self.isImaginary(weight-iSum))
                if(imaginary and eitherRootImaginary):
                    continue
                words1 = self.getWords(i)
                for word1 in words1:
                    for word2 in words2:
                        if(word1< word2):
                            (a,b) = (word1,word2)
                        else:
                            (a,b) = (word2,word1)
                        #FIXME:
                        if(not self.affine):
                            if(word.letterListCmp(a.string, maxWord.string) < 0):
                                continue
                            maxWord = a+b
                        if(self.affine and not imaginary):
                            if(np.all(combinations == np.array([1,1,1,1,1]))):
                                pass
                            #Checks to see if bracket is non-zero
                            if(a.height > 1 and word.letterListCmp(a.string[a.cofactorizationSplit:],b.string) < 0):
                                continue
                            newWord = a+b
                            if word.letterListCmp(newWord.string,maxWord.string) <= 0:
                                continue
                            if eitherRootImaginary and not self.eBracket(newWord):
                                continue
                            newWord.cofactorizationSplit = a.height
                            maxWord = newWord
                        if(imaginary):
                            if(a.height > 1 and word.letterListCmp(a.string[a.cofactorizationSplit:],b.string) < 0):
                                continue
                            potentialOptions.append(a+b)
                            continue
            if(self.affine):
                kDelta+= self.delta
        if not imaginary:
            maxWord.cofactorizationSplit = len(self.costandardFactorization(maxWord)[0])
            self.weightToWordDictionary[combinations.tobytes()] = [maxWord]
        else:
            potentialOptions.sort(reverse=True)
            matrix = np.zeros((self.n,self.n), dtype = int)
            liPotentialOptions = []
            index = 0
            row = 0
            while(row < self.n):
                potentialOptions[index].hs = self.hBracket(potentialOptions[index])
                matrix[row] = potentialOptions[index].hs
                if(np.linalg.matrix_rank(matrix) == row+1):
                    liPotentialOptions.append(potentialOptions[index])
                    row+=1
                index += 1
            self.weightToWordDictionary[combinations.tobytes()] = liPotentialOptions
    def getBaseWeights(self):
        return np.array(self.baseWeights)
    def getWordsByBase(self):
        returnarr = []
        for i in self.getBaseWeights():
            returnarr.append(np.array(self.getAffineWords(i)))
        return np.array(returnarr)
    def getMonotonicity(self, comb,deltaIndex = 0):
        words = self.getAffineWords(comb)
        for j in range(1,len(words)):
            monotonicity = 0
            if(words[j-1] < words[j]):
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
        wordsByLength = sorted(list(self.weightToWordDictionary.values()),key=lambda x:x[0].heights)
        for wordIndex in range(1,len(wordsByLength)):
            for sumWord in wordsByLength[wordIndex]:
                for alphaWords in wordsByLength[:wordIndex]:
                    for alphaWord in alphaWords:
                        for betaWord in self.getWords(sumWord.weights - alphaWord.weights):
                            if( betaWord<sumWord == alphaWord < sumWord):
                                exceptions.append((betaWord,alphaWord))
        return exceptions
    def TypeADelta(n:int):
        return np.ones(n+1,dtype=int)
    def TypeBDelta(n:int):
        arr = np.repeat(2,n+1)
        arr[-1] = 1
        arr[0] = 1
        return arr
    def TypeCDelta(n:int):
        delta = np.repeat(2,n+1)
        delta[-1] = 1
        delta[-2] = 1
        return delta
    def TypeDDelta(n:int):
        delta = np.repeat(2,n+1)
        delta[-1] =1
        delta[-2] = 1
        delta[-3] = 1
        delta[0] = 1
        return delta
    def TypeFDelta():
        return np.array([2,3,4,2,1],dtype=int)
    def TypeGDelta():
        return np.array([2,3,1],dtype=int)