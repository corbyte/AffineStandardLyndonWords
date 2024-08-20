import numpy as np
class letter:
    index:int
    rootIndex:int
    def __init__(self,rootIndex:int=0):
        self.rootIndex = rootIndex
    def set_index(self, i:int):
        self.index = i
    def __str__(self):
        return str(self.rootIndex)
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
    def letterListCmp(first,second):
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
    def strictLetterListCmp(first,second):
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
        return 0
    def listCostFac(list):
        smallestIndex = len(list)-1
        for i in range(len(list)-1,0,-1):
            if(word.letterListCmp(list[smallestIndex:],list[i:]) > 0):
                smallestIndex = i
        return smallestIndex        
    def noCommas(self):
        return ''.join(str(i) for i in self.string)
class letterOrdering:
    def __init__(self, letterOrdering):
        letterOrdering = [letter(i) for i in letterOrdering]
        self.order:list[letter] = letterOrdering
        for i in range(len(letterOrdering)):
            self.order[i].index = i
    def __len__(self):
        return len(self.order)
    def __getitem__(self,index) -> letter:
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
    def toLetterList(self) -> np.ndarray:
        lst = np.zeros(len(self.order),dtype=object)
        for let in self.order:
            lst[let.rootIndex] = let
        return lst
    def toOrderedList(self) -> np.ndarray:
        return np.array(self.order,dtype=object)
class rootSystem:
    def eBracket(self, bracketWord:word):            
        (A,B) = self.costfac(bracketWord)
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
        weights = re.weights - (self.delta * re.weights[0])
        return np.dot(weights[1:] ,(self.cartan_matrix @im.hs)) != 0
    def hBracket(self,word:word) -> np.array:
        a = self.costfac(word)[0]
        if(a is None):
            return np.zeros(self.n,dtype=int)
        word.cofactorizationSplit = a.height
        newA = (a.weights - (self.delta *a.weights[0]))
        if(np.any(newA < 0)):
            return -newA[1:]
        return newA[1:]
    def listHBracketing(self,letterList) -> bool:
        letterlist = letterList.string[:word.listCostFac(letterList)]
        weights = self.letterListToWeights(letterlist)
        newA = (weights - (self.delta *weights[0]))
        if(np.any(newA < 0)):
            return -newA[1:]
        return newA[1:]
    def listEBracketing(self,letterList):
        costFacIndex = word.listCostFac(letterList)
        A = letterList.string[costFacIndex:]
        B = letterList.string[:costFacIndex]
        if(A is None):
            return False
        if(self.isImaginary(len(A))):
            re = B
            im = A
        elif(self.isImaginary(len(B))):
            re = A
            im = B
        else:
            return True
        hs = self.listHBracketing(im)
        weights = re.weights - (self.delta * re.weights[0])
        return np.dot(weights[1:] ,(self.cartan_matrix @hs)) != 0
    def __init__(self, ordering,type:str):
        self.type = type.upper()
        if( len(self.type) != 1 or self.type < 'A' or self.type > 'G' ):
            raise ValueError('Type is invalid')
        self.n = len(ordering)-1
        self.ordering:letterOrdering = letterOrdering(ordering)
        self.weightToWordDictionary:dict = {}
        self.minWord:word = None
        def weightsGeneration(letter):
            arr = np.zeros(len(self.ordering),dtype=int)
            arr[letter.rootIndex] = 1
            return arr
        for i in [word([i],weightsGeneration(i)) for i in self.ordering.order]:
            if(self.minWord is None):
                self.minWord = i
            elif(self.minWord > i):
                self.minWord = i
            i.cofactorizationSplit = 0
            self.weightToWordDictionary[i.weights.tobytes()] = [i]
        if(self.type == 'A'):
            self.baseWeights = rootSystem.getAWeights(self.n)
        elif(self.type == 'B'):
            self.baseWeights = rootSystem.getBWeights(self.n)
        elif(self.type == 'C'):
            self.baseWeights = rootSystem.getCWeights(self.n)
        elif(self.type =='D'):
            self.baseWeights = rootSystem.getDWeights(self.n)
        elif(self.type == 'E'):
            self.baseWeights = rootSystem.getEWeights(self.n)
        elif(self.type == 'F'):
            self.baseWeights = rootSystem.getFWeights()
        elif(self.type == 'G'):
            self.baseWeights = rootSystem.getGWeights()
        self.__adjDict = self.__genAdjacencyDict()
        self.numberOfBaseWeights = len(self.baseWeights)
        if(self.type == 'C' and self.n == 2):
            self.cartan_matrix = np.array([
                [2,-2],
                [-1,2]
            ])
        else:
            self.cartan_matrix = rootSystem.getCartanMatrix(self.type,self.n)
        if(self.type == 'A'):
            self.delta = rootSystem.TypeADelta(self.n)
        elif (self.type == 'B'):
            self.delta = rootSystem.TypeBDelta(self.n) 
        elif(self.type == 'C'):
            self.delta = rootSystem.TypeCDelta(self.n)
        elif(self.type == 'D'):
            self.delta = rootSystem.TypeDDelta(self.n)
        elif(self.type == 'E'):
            self.delta = rootSystem.TypeEDelta(self.n)
        elif(self.type == 'F'):
            self.delta = rootSystem.TypeFDelta()
        elif(self.type == 'G'):
            self.delta = rootSystem.TypeGDelta()
        self.delta.flags.writeable = False
        self.deltaHeight = sum(self.delta)
    def getAWeights(n):
        size = n + 1
        arr = []
        for length in range(1,n+1):
            for start in range(0,n - length + 1):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k+1] = 1
                arr.append(comb)
        rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeADelta(n))
        arr.sort(key = sum)
        return np.array(arr) 
    def getBWeights(n):
        size = n+1
        arr = []
        for i in range(1,n+1):
            comb = np.zeros(size,dtype=int)
            comb[i] = 1
            arr.append(comb)
        for length in range(2,2*n):
            #i
            if(length <= n):
                comb = np.zeros(size,dtype=int)
                for i in range(n-length,n):
                    comb[i+1] = 1
                arr.append(comb)
                if(length != n):
                    #ei - ej
                    for start in range(0,n - length):
                        comb = np.zeros(size,dtype=int)
                        for k in range(start,start+length):
                            comb[k+1] = 1
                        arr.append(comb)
            #ei + ej
            if(length >= 3):
                comb = np.zeros(size,dtype=int)
                comb[n] = 2
                for i in range(n-min(length-1,n),n-1):
                    comb[i+1] = 1
                oneIndex = n-min(length-1,n)
                twoIndex = n-1
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex+1] = 2
                while(oneIndex < twoIndex):
                    arr.append(np.array(comb,dtype=int))
                    twoIndex -= 1
                    comb[twoIndex+1] = 2
                    comb[oneIndex+1] = 0
                    oneIndex+=1
        rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeBDelta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def getCWeights(n):
        size = n+1
        arr = []
        for i in range(n):
            comb = np.zeros(size,dtype=int)
            comb[i+1] = 1
            arr.append(comb)
        for length in range(2,2*n):
            if(length <= 2*n -2):
                #i+j
                comb = np.zeros(size,dtype=int)
                i = n-1
                while i >= 0 and n-i <= length:
                    comb[i+1] = 1
                    i -= 1
                oneIndex = i+1
                twoIndex = n-1
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex+1] = 2
                while(oneIndex < twoIndex):
                    arr.append(np.copy(comb))
                    twoIndex -= 1
                    comb[twoIndex+1] = 2
                    comb[oneIndex+1] = 0
                    oneIndex+=1
            #i-j
            if(length < n):
                for start in range(0,n-length):
                    comb = np.zeros(size,dtype=int)
                    for k in range(start,start+length):
                        comb[k+1] =1
                    arr.append(comb)
            #2i
            if(length % 2 == 1):
                comb=np.zeros(size,dtype=int)
                comb[n] = 1
                for k in range(n-2,n-2-length//2,-1):
                    comb[k+1] = 2
                arr.append(comb)
        rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeCDelta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def getDWeights(n):
        size = n+1
        arr = []
        for i in range(n):
            comb = np.zeros(size,dtype=int)
            comb[i+1] = 1
            arr.append(comb)
        for length in range(2,2*n-2):
            #i-j
            for start in range(0,n - length):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k+1] = 1
                arr.append(comb)
            #i+j
            if(length < n):
                comb=np.zeros(size,dtype=int)
                comb[n] =1
                for i in range(n-1-length,n-2):
                    comb[i+1] = 1
                arr.append(np.copy(comb))
            if(length >= 3):
                comb=np.zeros(size,dtype=int)
                comb[n] = 1
                comb[n-1] = 1
                for i in range(n-min(length,n),n-2):
                    comb[i+1] = 1
                oneIndex = n-min(length,n)
                twoIndex = n-2
                while(sum(comb) < length):
                    twoIndex -= 1
                    comb[twoIndex+1] = 2
                while(oneIndex < twoIndex):
                    arr.append(np.copy(comb))
                    twoIndex -= 1
                    comb[twoIndex+1] = 2
                    comb[oneIndex+1] = 0
                    oneIndex+=1
        rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeDDelta(n))
        arr.sort(key = sum)
        return arr
    def getEWeights(n,affine:bool=True):
        arr = []
        weights  = []
        if(n==6):
            weights = [
                [1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
                [1,1,0,0,0,0],[0,1,1,0,0,0],[0,0,1,1,0,0],[0,0,1,0,0,1],[0,0,0,1,1,0],[1,1,1,0,0,0],
                [0,1,1,1,0,0],[0,0,1,1,1,0],[0,1,1,0,0,1],[0,0,1,1,0,1],[0,1,1,1,0,1],[1,1,1,1,0,0],
                [1,1,1,0,0,1],[0,1,1,1,1,0],[0,0,1,1,1,1],[1,1,1,1,0,1],[0,1,2,1,0,1],[1,1,1,1,1,0],
                [0,1,1,1,1,1],[1,1,1,1,1,1],[1,1,2,1,0,1],[0,1,2,1,1,1],[1,1,2,1,1,1],[1,2,2,1,0,1],
                [0,1,2,2,1,1],[1,1,2,2,1,1],[1,2,2,1,1,1],[1,2,2,2,1,1],[1,2,3,2,1,1],[1,2,3,2,1,2]
            ]
        elif(n==7):
            weights = [
                [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1],[1,1,0,0,0,0,0],[0,1,1,0,0,0,0],[0,0,1,1,0,0,0],[0,0,0,1,1,0,0],[0,0,0,1,0,0,1],
                [0,0,0,0,1,1,0],[1,1,1,0,0,0,0],[0,1,1,1,0,0,0],[0,0,1,1,1,0,0],[0,0,0,1,1,1,0],[0,0,1,1,0,0,1],
                [0,0,0,1,1,0,1],[0,0,1,1,1,0,1],[1,1,1,1,0,0,0],[0,1,1,1,1,0,0],[0,1,1,1,0,0,1],[0,0,1,1,1,1,0],
                [0,0,0,1,1,1,1],[0,1,1,1,1,0,1],[1,1,1,1,1,0,0],[1,1,1,1,0,0,1],[0,0,1,2,1,0,1],[0,1,1,1,1,1,0],
                [0,0,1,1,1,1,1],[0,1,1,1,1,1,1],[1,1,1,1,1,1,0],[1,1,1,1,1,0,1],[0,1,1,2,1,0,1],[0,0,1,2,1,1,1],
                [1,1,1,1,1,1,1],[0,1,1,2,1,1,1],[0,1,2,2,1,0,1],[1,1,1,2,1,0,1],[0,0,1,2,2,1,1],[1,1,2,2,1,0,1],
                [1,1,1,2,1,1,1],[0,1,1,2,2,1,1],[0,1,2,2,1,1,1],[1,2,2,2,1,0,1],[0,1,2,2,2,1,1],[1,1,2,2,1,1,1],
                [1,1,1,2,2,1,1],[1,1,2,2,2,1,1],[1,2,2,2,1,1,1],[0,1,2,3,2,1,1],[1,1,2,3,2,1,1],[1,2,2,2,2,1,1],
                [0,1,2,3,2,1,2],[1,2,2,3,2,1,1],[1,1,2,3,2,1,2],[1,2,3,3,2,1,1],[1,2,2,3,2,1,2],[1,2,3,3,2,1,2],
                [1,2,3,4,2,1,2],[1,2,3,4,3,1,2],[1,2,3,4,3,2,2]
            ]
        else:
            raise Exception("Weights not implemented")
        for i in weights:
            if(affine):
                i = np.insert(i,0,0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeEDelta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def getFWeights(affine:bool=True):
        arr = []
        for i in [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,1],[0,1,1,1],
                  [0,1,2,2],[0,1,2,1],[1,1,1,1],[1,1,2,1],[1,2,2,1],[1,2,3,1],
                  [0,1,2,0],[1,1,2,0],[1,2,2,0],[1,2,2,2],[1,1,1,0],[0,1,1,0],
                  [1,3,4,2],[2,3,4,2],[1,1,2,2],[1,2,3,2],[1,1,0,0],[1,2,4,2]]:
            if(affine):
                i = np.insert(i,0,0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeFDelta())
        arr.sort(key = sum)
        return np.array(arr)
    def getGWeights(affine:bool=True)-> np.array:
        arr = []
        for i in [[1,0],[0,1],[1,1],[1,2],[1,3],[2,3]]:
            if(affine):
                i = np.insert(i,0,0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__genAffineBaseWeights(arr,rootSystem.TypeGDelta())
        arr.sort(key = sum)
        return np.array(arr)
    def __genAffineBaseWeights(arr,delta:np.array):
        for i in arr:
            if(i[0] == 1):
                break
            arr.append(delta - i)
        arr.append(delta)
    def costfac(self,wordToFactor:word):
        if(wordToFactor.height == 1):
            return (wordToFactor,None)
        weight = np.copy(wordToFactor.weights)
        weight[wordToFactor.string[0].rootIndex] -= 1
        splitLetter = None
        for i in self.ordering:
            if(weight[i.rootIndex] != 0):
                splitLetter = i
                break
        weight[wordToFactor.string[0].rootIndex] += 1
        for i in range(1,wordToFactor.height):
            weight[wordToFactor.string[i-1].rootIndex] -= 1
            if(wordToFactor.string[i].index != splitLetter.index):
                continue
            rightWords = self.__getWords(weight)
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
            leftWords = self.__getWords(wordToFactor.weights-weight)    
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
    def __getWords(self, combination:np.array):
        return self.weightToWordDictionary.get(combination.tobytes(),[])
    def getWords(self, combination):
        if(self.containsWeight(combination)):
            ret = self.__getWords(np.array(combination, dtype=int))
            if(len(ret) == 0):
                self.generateUptoHeight(sum(combination))
                ret = self.__getWords(np.array(combination, dtype=int))
            return ret
        return []
    def getAffineWords(self,weight):
        matches = []
        k=0
        newWord = self.__getWords(weight + k*self.delta)
        while len(newWord) > 0:
            matches.extend(newWord)
            k+=1
            newWord=self.__getWords(weight + k*self.delta)
        return matches
    def isImaginary(self,height:int):
        return height % self.deltaHeight == 0
    def __genWord(self, combinations:np.array):
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
                words2 = self.__getWords(j)
                if(len(words2) == 0):
                    validBase[baseWordIndex] = False
                    continue
                lengthChecked = weight - iSum
                eitherRootImaginary = (self.isImaginary(iSum) or self.isImaginary(weight-iSum))
                if(imaginary and eitherRootImaginary):
                    continue
                words1 = self.__getWords(i)
                for word1 in words1:
                    for word2 in words2:
                        if(word1< word2):
                            (a,b) = (word1,word2)
                        else:
                            (a,b) = (word2,word1)
                        if(not imaginary):
                            #Checks to see if bracket is non-zero
                            newWord = a+b
                            if word.letterListCmp(newWord.string,maxWord.string) <= 0:
                                continue
                            if eitherRootImaginary and not self.eBracket(newWord):
                                continue
                            maxWord = newWord
                        if(imaginary):
                            if(a.height > 1 and word.letterListCmp(a.string[a.cofactorizationSplit:],b.string) < 0):
                                continue
                            potentialOptions.append(a+b)
                            continue
            kDelta+= self.delta
        if not imaginary:
            maxWord.cofactorizationSplit = len(self.costfac(maxWord)[0])
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
            returnarr.append((self.__getWords(i)[0].weights,monotonicity))
        return np.array(returnarr, dtype=object)
    def checkConvexity(self):
        exceptions = []
        wordsByLength = sorted(list(self.weightToWordDictionary.values()),key=lambda x:x[0].height)
        for wordIndex in range(1,len(wordsByLength)):
            for sumWord in wordsByLength[wordIndex]:
                for alphaWords in wordsByLength[:wordIndex]:
                    for alphaWord in alphaWords:
                        for betaWord in self.__getWords(sumWord.weights - alphaWord.weights):
                            if( betaWord<sumWord == alphaWord < sumWord):
                                exceptions.append((betaWord,alphaWord))
        return exceptions
    def getDecompositions(self,weights):
        if(weights is not np.array):
            weights = np.array(weights)
        returnarr = []
        delta = 0
        while(delta*self.deltaHeight< sum(weights)):
            for i in self.baseWeights:
                if len(self.__getWords(weights-i - delta*self.delta)) > 0:
                    returnarr.append((i,weights-i-delta*self.delta))
            delta+= 1 
        return returnarr
    def getPotentialWords(self,weights):
        decomps = self.getDecompositions(weights)
        arr =[]
        for i in decomps:
            for j in self.__getWords(i[0]):
                for k in self.__getWords(i[1]):
                    if(j < k):
                        arr.append(j+k)
                    else:
                        arr.append(k+j)
        return arr
    def TypeADelta(n:int):
        return np.ones(n+1,dtype=int)
    def TypeBDelta(n:int):
        arr = np.repeat(2,n+1)
        arr[0] = 1
        arr[1] = 1
        return arr
    def TypeCDelta(n:int):
        delta = np.repeat(2,n+1)
        delta[0] = 1
        delta[-1] = 1
        return delta
    def TypeDDelta(n:int):
        delta = np.repeat(2,n+1)
        delta[0] =1
        delta[-1] = 1
        delta[-2] = 1
        delta[1] = 1
        return delta
    def TypeEDelta(n:int):
        if(n == 6):
            return np.array([1,1,2,3,2,1,2],dtype=int)
        if(n == 7):
            return np.array([1,1,2,3,4,3,2,2],dtype=int)
    def TypeFDelta():
        return np.array([1,2,3,4,2],dtype=int)
    def TypeGDelta():
        return np.array([1,2,3],dtype=int)
    def printFormat(words, formatfunc):
        for word in words:
            print(formatfunc(word))
    def deltaFormat(self,word):
        if(word is None):
            return "None"
        deltaWeight = word.height//self.deltaHeight
        base = word.weights - deltaWeight*self.delta
        return f"{base} + {deltaWeight}d"
    def SLDeltaFormat(self,word):
        retstr = word.noCommas()
        deltaWords = self.__getWords(self.delta)
        for i in range(len(deltaWords)):
            retstr = retstr.replace(deltaWords[i].noCommas(),f"SL_{{{i+1}}}(d)")
        return retstr
    def parseToDeltaFormat(self,parseWord:word):
        retarr = []
        deltaWords = self.__getWords(self.delta)
        stack = []
        for letter in parseWord.string:
            stack.append(letter)
            if(len(stack) >= self.deltaHeight):
                for i in range(len(deltaWords)):
                    deltaWord = deltaWords[i]
                    if(word.letterListCmp(deltaWord.string,stack[-self.deltaHeight:]) == 0):
                        string = ""
                        for stackLetter in stack[:-self.deltaHeight]:
                            string+=str(stackLetter)
                        if(string != ""):
                            retarr.append(string)
                        if(len(retarr) > 0 and type(retarr[-1]) is list and retarr[-1][0] == i+1):
                            retarr[-1][1]+=1
                        else:
                            retarr.append([i+1,1])
                        stack = []
                        break
        string=""
        for j in range(len(stack)):
            string+=str(stack[j])
        if(len(string) > 0): 
            retarr.append(string)
        return retarr
    def getCartanMatrix(type:str,n:int):
        if(n <= 0):
            raise ValueError("Invalid Parameters")
        type=type.upper()
        if(type == 'A'):
            mat = 2*np.eye(n,dtype=int)
            for i in range(0,n-1):
                mat[i][i+1] = -1
                mat[i+1][i] = -1
            return mat
        elif(type == 'B'):
            mat = 2*np.eye(n,dtype=int)
            for i in range(0,n-2):
                mat[i][i+1] = -1
                mat[i+1][i] = -1
            mat[-1][-2] = -2
            mat[-2][-1] = -1
            return mat
        elif(type == 'C'):
            mat = 2*np.eye(n,dtype=int)
            for i in range(0,n-2):
                mat[i][i+1] = -1
                mat[i+1][i] = -1
            mat[-1][-2] = -1
            mat[-2][-1] = -2
            return mat
        elif(type == 'D'):
            mat = 2*np.eye(n,dtype=int)
            for i in range(0,n-2):
                mat[i][i+1] = -1
                mat[i+1][i] = -1
            mat[-1][-3] = -1
            mat[-3][-1] = -1
            return mat
        elif(type == 'E'):
            if(n >= 6 and n <= 8):
                mat = 2*np.eye(n,dtype=int)
                for i in range(0,n-2):
                    mat[i][i+1] = -1
                    mat[i+1][i] = -1 
                mat[-1][-4] = -1
                mat[-4][-1] = -1
                return mat
        elif(type == 'F'):
            if(n == 4):
                mat = 2*np.eye(n,dtype=int)
                for i in range(0,n-1):
                    mat[i][i+1] = -1
                    mat[i+1][i] = -1
                mat[1][2] = -2
                return mat
        elif(type == 'G'):
            if(n == 2):
                return np.array([[2,-3],[-1,2]], dtype=int)
        raise ValueError("Invalid parameters")
    def getPeriodicity(self, simpleRoot,slIndex:int = 0) -> int:
        factors = []
        for i in self.getAffineWords(simpleRoot):
            tempArr = []
            for k in self.costfac(i):
                if(k is None):
                    tempArr.append([np.zeros(len(i),dtype=int),0])
                else:
                    imaginaryIndex=0
                    if(self.isImaginary(k.height)):
                        imaginaryIndex = slIndex+1
                    tempArr.append([k.weights - (k.height//self.deltaHeight)*self.delta,k.height//self.deltaHeight,imaginaryIndex])
            factors.append(tempArr)
        repeat = 1
        if(self.isImaginary(sum(simpleRoot))):
            repeat = self.n
        strings = factors[slIndex::repeat]
        for width in range(1,len(strings)//2):
            for windowStart in range(1,(len(strings)-2*width)):
                countEqual = 0
                for i in range(width):
                    arr1 = np.array([strings[windowStart+i + width*j][0][0] for j in range((len(strings)-(windowStart+i))//width)])
                    arr2 = np.array([strings[windowStart+i + width*j][1][0] for j in range((len(strings)-(windowStart+i))//width)])
                    if (arr1 == arr1[0]).all() and (arr2 == arr2[0]).all():
                        countEqual+= 1
                    else:
                        break
                if(countEqual == width):
                    return width
    def generateUptoHeight(self,height:int):
        k=0
        while True:
            for base in self.baseWeights:
                weight = base + k*self.delta
                if(sum(weight) > height):
                    return
                if len(self.__getWords(weight)) == 0:
                    self.__genWord(weight)
            k += 1
    def containsWeight(self,weights):
        if(len(weights) != self.n+1):
            return False
        weights = np.array(weights,dtype=int)
        weights -= ((sum(weights)-1)//self.deltaHeight) * self.delta
        return np.any(np.all(self.baseWeights[:] == weights,axis=1))
    def letterListToWeights(self,letterList):
        arr = np.zeros(self.n + 1,dtype=int)
        for l in letterList:
            arr[l.rootIndex] += 1
        return arr
    def __combineCurrentWords(currentWords,index1:int,index2:int) -> list:
        if(index1 < 0 or index2 < 0):
            raise ValueError("index 1 or 2 should be nonnegative numbers")
        twoGreater = index2 > index1
        word1 = currentWords[index1][0]
        word2 = currentWords[index2][0]
        if(rootSystem.__decrementList(currentWords,index1) and index1 < index2):
            index2 -= 1
        rootSystem.__decrementList(currentWords,index2)
        if(twoGreater):
            newWord = word2 + word1
        else:
            newWord = word1 + word2
        rootSystem.__addToList(currentWords,newWord)
        return currentWords
    def __decrementList(currentList, index:int) -> bool:
        if(currentList[index][1] == 1):
            currentList.pop(index)
            return True
        else:
            currentList[index][1] -= 1
            return False
    def __addToList(currentWords,w:word) -> bool:
        for i in range(len(currentWords)):
            if(word.letterListCmp(currentWords[i][0],w) == 0):
                currentWords[i][1] += 1
                return False
            if(word.letterListCmp(currentWords[i][0],w) < 0):
                currentWords.insert(i,[w,1])
                return True
        return False
    def __nextSmallest(self,index,currentList,excluded:set = set()):
        currentWord = currentList[index][0].string
        weight = currentList[index][0].weights
        rootSystem.__decrementList(currentList,index)
        letterWeight = np.zeros(self.n+1,dtype=int)
        removedLetter = currentWord[-1]
        letterWeight[removedLetter.rootIndex] = 1
        rootSystem.__addList(currentList,word([removedLetter],letterWeight))
        currentWord = currentWord[:-1]
        while True:
            flag = False
            for i in range(len(currentList),index):
                if(currentList[i][0] > removedLetter):
                    continue
                if(len(self.getWords(currentWords[-1][0].weights + currentWords[i][0].weights)) != 0
                    and currentList[i][0] not in excluded
                    ):
                        currentWords = rootSystem.__combineCurrentWords(currentWords,i,len(currentWords)-1)
                        foundFlag = True
                        break
            if(not flag):
                break                                           
        return currentList
    def SLWordAlgo(self,weightsToGenerate) -> list:
        if(sum(weightsToGenerate) > self.deltaHeight and self.isImaginary(sum(weightsToGenerate))):
            return []
        currentWords = []
        returnWord = None
        currentWeight = np.array(self.n+1,dtype=int)
        weightsToGenerate = np.array(weightsToGenerate,dtype=int)
        #currentWeight = np.zeros(self.n + 1, dtype=int)
        for i in range(len(self.ordering)-1,-1,-1):
            if(weightsToGenerate[self.ordering[i].rootIndex] > 0): 
                arr = np.zeros(self.n+1,dtype=int)
                arr[self.ordering[i].rootIndex] = 1 
                currentWords.append([word([self.ordering[i]],arr),weightsToGenerate[self.ordering[i].rootIndex]])
        while True:
            if(len(currentWords) and self.isImaginary(sum(currentWords[0][0].weights))):
                self.__nextSmallest(0,currentWords)
            if(len(currentWords) == 0):
                return returnWord
            for i in range(currentWords[-1][1]):
                foundFlag = False
                for possibleAppendInd in range(len(currentWords)):
                    if(len(self.getWords(currentWords[-1][0].weights + currentWords[possibleAppendInd][0].weights)) != 0):
                        currentWords = rootSystem.__combineCurrentWords(currentWords,possibleAppendInd,len(currentWords)-1)
                        foundFlag = True
                        break
                if(not foundFlag):
                    if(word.listEBracketing(currentWords[-1][0],self)):
                        if(returnWord is None):
                            returnWord = currentWords[-1][0]
                        else:
                            returnWord = returnWord + currentWords[-1][0]
                        rootSystem.__decrementList(currentWords,-1)
                    else:
                        self.__nextSmallest(len(currentWords-1),currentWords)
if(__name__ == "__main__"):
    F4 = rootSystem([0,2,1,3,4],"F",1)
    F4.SLWordAlgo([0, 2, 3, 4, 2])