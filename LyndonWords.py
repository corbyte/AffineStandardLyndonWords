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
        return word.letter_list_cmp(self.string,other.string) < 0
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
    def letter_list_cmp(first,second):
        lFirst = len(first)
        lSecond = len(second)
        if(lFirst < lSecond):
            minLen = lFirst
        else:
            minLen = lSecond
        for i in range(minLen):
            if(type(first[i])is not letter or type(second[i]) is not letter):
                raise ValueError("List contains non-letter object")
            if(first[i].index > second[i].index):
                return 1
            if(first[i].index < second[i].index):
                return -1
        if(lFirst < lSecond):
            return -1
        if(lFirst > lSecond):
            return 1
        return 0
    def strict_letter_list_cmp(first,second):
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
    def list_cost_fac(list):
        smallestIndex = len(list)-1
        for i in range(len(list)-1,0,-1):
            if(word.letter_list_cmp(list[smallestIndex:],list[i:]) > 0):
                smallestIndex = i
        return smallestIndex        
    def no_commas(self):
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
    def to_letter_list(self) -> np.ndarray:
        lst = np.zeros(len(self.order),dtype=object)
        for let in self.order:
            lst[let.rootIndex] = let
        return lst
    def to_ordered_list(self) -> np.ndarray:
        return np.array(self.order,dtype=object)
class rootSystem:
    def e_bracket(self, bracketWord:word):      
        if(len(bracketWord) == 1):
            return True      
        (A,B) = self.costfac(bracketWord)
        if(A is None):
            return False
        if(self.is_imaginary_height(A.height)):
            re = B
            im = A
        elif(self.is_imaginary_height(B.height)):
            re = A
            im = B
        else:
            return True
        weights = re.weights - (self.delta * re.weights[0])
        return np.dot(im.hs ,(self.sym_matrix @ weights[1:])) != 0
    def h_bracket(self,word:word) -> np.array:
        a = self.costfac(word)[0]
        if(a is None):
            return np.zeros(self.n,dtype=int)
        word.cofactorizationSplit = a.height
        newA = (a.weights - (self.delta *a.weights[0]))
        if(np.any(newA < 0)):
            return -newA[1:]
        return newA[1:]
    def list_h_bracketing(self,letterList) -> bool:
        letterlist = letterList[:word.list_cost_fac(letterList)]
        weights = self.letter_list_to_weights(letterlist)
        newA = (weights - (self.delta *weights[0]))
        if(np.any(newA < 0)):
            return -newA[1:]
        return newA[1:]
    def list_e_bracketing(self,letterList):
        costFacIndex = word.list_cost_fac(letterList)
        A = letterList[:costFacIndex]
        B = letterList[costFacIndex:]
        if(A is None):
            return False
        if(self.is_imaginary_height(len(A))):
            re = B
            im = A
        elif(self.is_imaginary_height(len(B))):
            re = A
            im = B
        else:
            return True
        hs = self.list_h_bracketing(im)
        realWeights = self.letter_list_to_weights(re)
        weights = realWeights - (self.delta * realWeights[0])
        return np.dot(weights[1:] ,(self.sym_matrix @hs)) != 0
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
            self.baseWeights = rootSystem.A_weights(self.n)
        elif(self.type == 'B'):
            self.baseWeights = rootSystem.B_weights(self.n)
        elif(self.type == 'C'):
            self.baseWeights = rootSystem.C_weights(self.n)
        elif(self.type =='D'):
            self.baseWeights = rootSystem.D_weights(self.n)
        elif(self.type == 'E'):
            self.baseWeights = rootSystem.E_weights(self.n)
        elif(self.type == 'F'):
            self.baseWeights = rootSystem.F_weights()
        elif(self.type == 'G'):
            self.baseWeights = rootSystem.G_weights()
        self.numberOfBaseWeights = len(self.baseWeights)
        if(self.type == 'C' and self.n == 2):
            self.cartan_matrix = np.array([
                [2,-2],
                [-1,2]
            ])
        else:
            self.cartan_matrix = rootSystem.get_cartan_matrix(self.type,self.n)
        if(self.type == 'A'):
            self.delta = rootSystem.A_delta(self.n)
        elif (self.type == 'B'):
            self.delta = rootSystem.B_delta(self.n) 
        elif(self.type == 'C'):
            self.delta = rootSystem.C_delta(self.n)
        elif(self.type == 'D'):
            self.delta = rootSystem.D_delta(self.n)
        elif(self.type == 'E'):
            self.delta = rootSystem.E_delta(self.n)
        elif(self.type == 'F'):
            self.delta = rootSystem.F_delta()
        elif(self.type == 'G'):
            self.delta = rootSystem.G_delta()
        self.delta.flags.writeable = False
        self.deltaHeight = sum(self.delta)
        self.vectors_norm2 = rootSystem.basis_vector_norm2(self.type,self.n)
        self.sym_matrix = self.get_sym_matrix()
    def A_weights(n):
        size = n + 1
        arr = []
        for length in range(1,n+1):
            for start in range(0,n - length + 1):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k+1] = 1
                arr.append(comb)
        rootSystem.__gen_affine_base_weights(arr,rootSystem.A_delta(n))
        arr.sort(key = sum)
        return np.array(arr) 
    def B_weights(n):
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
        rootSystem.__gen_affine_base_weights(arr,rootSystem.B_delta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def C_weights(n):
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
        rootSystem.__gen_affine_base_weights(arr,rootSystem.C_delta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def D_weights(n):
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
        rootSystem.__gen_affine_base_weights(arr,rootSystem.D_delta(n))
        arr.sort(key = sum)
        return arr
    def E_weights(n,affine:bool=True):
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
        elif(n==8):
            weights = [
                [1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],
                [1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,0,0,1],[0,0,0,0,0,1,1,0],[1,1,1,0,0,0,0,0],
                [0,1,1,1,0,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,1,1,0,0,1],[0,0,0,0,1,1,0,1],[0,0,0,1,1,1,0,1],[1,1,1,1,0,0,0,0],
                [0,1,1,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,0,0,1],[0,0,0,1,1,1,1,0],[0,0,0,0,1,1,1,1],[0,0,1,1,1,1,0,1],[1,1,1,1,1,0,0,0],[0,1,1,1,1,1,0,0],
                [0,1,1,1,1,0,0,1],[0,0,0,1,2,1,0,1],[0,0,1,1,1,1,1,0],[0,0,0,1,1,1,1,1],[0,0,1,1,1,1,1,1],[1,1,1,1,1,1,0,0],[1,1,1,1,1,0,0,1],[0,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,0,1],[0,0,1,1,2,1,0,1],[0,0,0,1,2,1,1,1],[0,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0],[0,0,1,1,2,1,1,1],[0,0,1,2,2,1,0,1],[1,1,1,1,1,1,0,1],
                [0,1,1,1,2,1,0,1],[0,0,0,1,2,2,1,1],[1,1,1,1,2,1,0,1],[0,1,1,2,2,1,0,1],[1,1,1,1,1,1,1,1],[0,1,1,1,2,1,1,1],[0,0,1,1,2,2,1,1],[0,0,1,2,2,1,1,1],
                [1,1,1,2,2,1,0,1],[0,1,2,2,2,1,0,1],[0,0,1,2,2,2,1,1],[0,1,1,2,2,1,1,1],[1,1,1,1,2,1,1,1],[0,1,1,1,2,2,1,1],[1,1,1,1,2,2,1,1],[1,1,2,2,2,1,0,1],
                [1,1,1,2,2,1,1,1],[0,1,1,2,2,2,1,1],[0,1,2,2,2,1,1,1],[0,0,1,2,3,2,1,1],[1,2,2,2,2,1,0,1],[0,1,1,2,3,2,1,1],[1,1,2,2,2,1,1,1],[0,1,2,2,2,2,1,1],
                [1,1,1,2,2,2,1,1],[0,0,1,2,3,2,1,2],[1,1,2,2,2,2,1,1],[1,1,1,2,3,2,1,1],[1,2,2,2,2,1,1,1],[0,1,2,2,3,2,1,1],[0,1,1,2,3,2,1,2],[0,1,2,3,3,2,1,1],
                [1,2,2,2,2,2,1,1],[0,1,2,2,3,2,1,2],[1,1,2,2,3,2,1,1],[1,1,1,2,3,2,1,2],[0,1,2,3,3,2,1,2],[1,1,2,3,3,2,1,1],[1,2,2,2,3,2,1,1],[1,1,2,2,3,2,1,2],
                [1,2,2,3,3,2,1,1],[1,1,2,3,3,2,1,2],[1,2,2,2,3,2,1,2],[0,1,2,3,4,2,1,2],[1,2,3,3,3,2,1,1],[1,1,2,3,4,2,1,2],[1,2,2,3,3,2,1,2],[0,1,2,3,4,3,1,2],
                [1,2,3,3,3,2,1,2],[1,2,2,3,4,2,1,2],[1,1,2,3,4,3,1,2],[0,1,2,3,4,3,2,2],[1,2,2,3,4,3,1,2],[1,2,3,3,4,2,1,2],[1,1,2,3,4,3,2,2],[1,2,3,4,4,2,1,2],
                [1,2,3,3,4,3,1,2],[1,2,2,3,4,3,2,2],[1,2,3,3,4,3,2,2],[1,2,3,4,4,3,1,2],[1,2,3,4,4,3,2,2],[1,2,3,4,5,3,1,2],[1,2,3,4,5,3,1,3],[1,2,3,4,5,3,2,2],
                [1,2,3,4,5,4,2,2],[1,2,3,4,5,3,2,3],[1,2,3,4,5,4,2,3],[1,2,3,4,6,4,2,3],[1,2,3,5,6,4,2,3],[1,2,4,5,6,4,2,3],[1,3,4,5,6,4,2,3],[2,3,4,5,6,4,2,3]
            ]
        else:
            raise ValueError("Please enter 6,7,8")
        for i in weights:
            if(affine):
                i = np.insert(i,0,0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__gen_affine_base_weights(arr,rootSystem.E_delta(n))
        arr.sort(key = sum)
        return np.array(arr)
    def F_weights(affine:bool=True):
        arr = []
        for i in [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,1],[0,1,1,1],
                  [0,1,2,2],[0,1,2,1],[1,1,1,1],[1,1,2,1],[1,2,2,1],[1,2,3,1],
                  [0,1,2,0],[1,1,2,0],[1,2,2,0],[1,2,2,2],[1,1,1,0],[0,1,1,0],
                  [1,3,4,2],[2,3,4,2],[1,1,2,2],[1,2,3,2],[1,1,0,0],[1,2,4,2]]:
            if(affine):
                i = np.insert(i,0,0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__gen_affine_base_weights(arr,rootSystem.F_delta())
        arr.sort(key = sum)
        return np.array(arr)
    def G_weights(affine:bool=True)-> np.array:
        arr = []
        for i in [[1,0],[0,1],[1,1],[1,2],[1,3],[2,3]]:
            if(affine):
                i = np.insert(i,0,0)
            arr.append(np.array(i,dtype=int))
        if(affine):
            rootSystem.__gen_affine_base_weights(arr,rootSystem.G_delta())
        arr.sort(key = sum)
        return np.array(arr)
    def __gen_affine_base_weights(arr,delta:np.array):
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
            rightWords = self.__get_words(weight)
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
            leftWords = self.__get_words(wordToFactor.weights-weight)    
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
    def __get_words(self, combination:np.array):
        return self.weightToWordDictionary.get(combination.tobytes(),[])
    def get_words(self, combination):
        if(self.contains_weight(combination)):
            ret = self.__get_words(np.array(combination, dtype=int))
            if(len(ret) == 0):
                self.generate_up_to_height(sum(combination))
                ret = self.__get_words(np.array(combination, dtype=int))
            return ret
        return []
    def get_affine_words(self,weight):
        matches = []
        k=0
        newWord = self.__get_words(weight + k*self.delta)
        while len(newWord) > 0:
            matches.extend(newWord)
            k+=1
            newWord=self.__get_words(weight + k*self.delta)
        return matches
    def is_imaginary_height(self,height:int):
        return height % self.deltaHeight == 0
    def __gen_word(self, combinations:np.array):
        weight = sum(combinations)
        if(weight == 1):
            return
        imaginary = self.is_imaginary_height(weight)
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
                words2 = self.__get_words(j)
                if(len(words2) == 0):
                    validBase[baseWordIndex] = False
                    continue
                lengthChecked = weight - iSum
                eitherRootImaginary = (self.is_imaginary_height(iSum) or self.is_imaginary_height(weight-iSum))
                if(imaginary and eitherRootImaginary):
                    continue
                words1 = self.__get_words(i)
                for word1 in words1:
                    for word2 in words2:
                        if(word1< word2):
                            (a,b) = (word1,word2)
                        else:
                            (a,b) = (word2,word1)
                        if(not imaginary):
                            #Checks to see if bracket is non-zero
                            newWord = a+b
                            if word.letter_list_cmp(newWord.string,maxWord.string) <= 0:
                                continue
                            if eitherRootImaginary and not self.e_bracket(newWord):
                                continue
                            maxWord = newWord
                        if(imaginary):
                            if(a.height > 1 and word.letter_list_cmp(a.string[a.cofactorizationSplit:],b.string) < 0):
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
                potentialOptions[index].hs = self.h_bracket(potentialOptions[index])
                matrix[row] = potentialOptions[index].hs
                if(np.linalg.matrix_rank(matrix) == row+1):
                    liPotentialOptions.append(potentialOptions[index])
                    row+=1
                index += 1
            self.weightToWordDictionary[combinations.tobytes()] = liPotentialOptions
    def get_words_by_base(self):
        returnarr = []
        for i in self.getBaseWeights():
            returnarr.append(np.array(self.get_affine_words(i)))
        return np.array(returnarr)
    def get_monotonicity(self, comb,deltaIndex = 0):
        words = self.get_affine_words(comb)
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
    def check_monotonicity(self, filter:{'All', 'Increasing', 'Decreasing','None'}="All"):
        returnarr = []
        for i in self.getBaseWeights()[:-1]:
            monotonicity = self.get_monotonicity(i)
            if(filter == 'None' and monotonicity != 0):
                continue
            if(filter == 'Increasing' and monotonicity != 1):
                continue
            if(filter == 'Decreasing' and monotonicity != -1):
                continue
            returnarr.append((self.__get_words(i)[0].weights,monotonicity))
        return np.array(returnarr, dtype=object)
    def check_convexity(self):
        exceptions = []
        wordsByLength = sorted(list(self.weightToWordDictionary.values()),key=lambda x:x[0].height)
        for wordIndex in range(1,len(wordsByLength)):
            for sumWord in wordsByLength[wordIndex]:
                for alphaWords in wordsByLength[:wordIndex]:
                    for alphaWord in alphaWords:
                        for betaWord in self.__get_words(sumWord.weights - alphaWord.weights):
                            if( betaWord<sumWord == alphaWord < sumWord):
                                exceptions.append((betaWord,alphaWord))
        return exceptions
    def get_decompositions(self,weights):
        if(weights is not np.array):
            weights = np.array(weights)
        returnarr = []
        delta = 0
        while(delta*self.deltaHeight< sum(weights)):
            for i in self.baseWeights:
                if len(self.__get_words(weights-i - delta*self.delta)) > 0:
                    returnarr.append((i,weights-i-delta*self.delta))
            delta+= 1 
        return returnarr
    def get_potential_words(self,weights):
        decomps = self.get_decompositions(weights)
        arr =[]
        for i in decomps:
            for j in self.__get_words(i[0]):
                for k in self.__get_words(i[1]):
                    if(j < k):
                        arr.append(j+k)
                    else:
                        arr.append(k+j)
        return arr
    def A_delta(n:int):
        return np.ones(n+1,dtype=int)
    def B_delta(n:int):
        arr = np.repeat(2,n+1)
        arr[0] = 1
        arr[1] = 1
        return arr
    def C_delta(n:int):
        delta = np.repeat(2,n+1)
        delta[0] = 1
        delta[-1] = 1
        return delta
    def D_delta(n:int):
        delta = np.repeat(2,n+1)
        delta[0] =1
        delta[-1] = 1
        delta[-2] = 1
        delta[1] = 1
        return delta
    def E_delta(n:int):
        if(n == 6):
            return np.array([1,1,2,3,2,1,2],dtype=int)
        if(n == 7):
            return np.array([1,1,2,3,4,3,2,2],dtype=int)
        if(n == 8):
            return np.array([1,2,3,4,5,6,4,2,3])
    def F_delta():
        return np.array([1,2,3,4,2],dtype=int)
    def G_delta():
        return np.array([1,2,3],dtype=int)
    def basis_vector_norm2(t:str, n:int) -> np.array:
        t = t.upper()
        if(n<= 0):
            raise ValueError("n must be positive")
        values = np.zeros(n,dtype=int)
        if(t == 'A'):
            for i in range(len(values)):
                values[i] = 2
            return values
        elif(t == 'B'):
            for i in range(len(values)-1):
                values[i] = 2
            values[-1] = 1
            return values
        elif( t== 'C'):
            for i in range(len(values)-1):
                values[i] = 2
            values[-1] = 4
            return values
        elif(t == 'D'):
            for i in range(len(values)-1):
                values[i] = 2
            return values
        elif(t == 'E'):
            if(n > 5 and n < 9):
                for i in range(len(values)):
                    values[i] = 2
                return values
            else:
                raise ValueError("Invaid n for Type E")
        elif(t == 'F'):
            if(n == 4):
                values[0] = 2
                values[1] = 2
                values[2] = 1
                values[3] = 1
                return values
            else:
                raise ValueError("Invalid n for Type F")
        elif(t == 'G'):
            if(n == 2):
                values[0] = 2
                values[1] = 3
                return values
            else:
                raise ValueError("Invalid n for Type G")
    def get_sym_matrix(self):
        diag = np.zeros((self.n,self.n),dtype=int)
        for i in range(self.n):
            diag[i,i] = self.vectors_norm2[i]
        return diag @ self.cartan_matrix
    def print_format(words, formatfunc):
        for word in words:
            print(formatfunc(word))
    def delta_format(self,word):
        retstr = word.noCommas()
        deltaWords = self.__get_words(self.delta)
        for i in range(len(deltaWords)):
            retstr = retstr.replace(deltaWords[i].noCommas(),f"SL_{{{i+1}}}(d)")
        return retstr
    def parse_to_delta_format(self,parseWord:word):
        retarr = []
        deltaWords = self.__get_words(self.delta)
        stack = []
        for letter in parseWord.string:
            stack.append(letter)
            if(len(stack) >= self.deltaHeight):
                for i in range(len(deltaWords)):
                    deltaWord = deltaWords[i]
                    if(word.letter_list_cmp(deltaWord.string,stack[-self.deltaHeight:]) == 0):
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
    def get_cartan_matrix(type:str,n:int):
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
            mat[-2][-1] = -1
            mat[-1][-2] = -2
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
                mat[2][1] = -2
                return mat
        elif(type == 'G'):
            if(n == 2):
                return np.array([[2,-3],[-1,2]], dtype=int)
        raise ValueError("Invalid parameters")
    def get_periodicity(self, simpleRoot,slIndex:int = 0,kdelta:int = 3) -> int:
        factors = []
        self.generate_up_to_delta(kdelta)
        for i in self.get_affine_words(simpleRoot):
            tempArr = []
            for k in self.costfac(i):
                if(k is None):
                    tempArr.append([np.zeros(len(i),dtype=int),0])
                else:
                    tempArr.append([k.weights - (k.height//self.deltaHeight)*self.delta,k.height//self.deltaHeight])
            factors.append(tempArr)
        repeat = 1
        if(self.is_imaginary_height(sum(simpleRoot))):
            repeat = self.n
        strings = factors[slIndex::repeat]
        while True:
            for width in range(1,len(strings)//2):
                for windowStart in range(1,(len(strings)-3*width)):
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
            added = 3
            for i in range(added):
                kdelta+=1
                w = self.get_words(simpleRoot + self.delta*kdelta)[slIndex]
                tempArr = []
                for k in self.costfac(w):
                    if(k is None):
                        tempArr.append([np.zeros(len(i),dtype=int),0])
                    else:
                        tempArr.append([k.weights - (k.height//self.deltaHeight)*self.delta,k.height//self.deltaHeight])
                strings.append(tempArr)
    def generate_up_to_height(self,height:int):
        k=0
        while True:
            for base in self.baseWeights:
                weight = base + k*self.delta
                if(sum(weight) > height):
                    return
                if len(self.__get_words(weight)) == 0:
                    self.__gen_word(weight)
            k += 1
    def generate_up_to_delta(self,k:int):
        self.generate_up_to_height(self.deltaHeight*k)
    def contains_weight(self,weights):
        if(len(weights) != self.n+1):
            return False
        weights = np.array(weights,dtype=int)
        weights -= ((sum(weights)-1)//self.deltaHeight) * self.delta
        return np.any(np.all(self.baseWeights[:] == weights,axis=1))
    def letter_list_to_weights(self,letterList):
        arr = np.zeros(self.n + 1,dtype=int)
        for l in letterList:
            arr[l.rootIndex] += 1
        return arr
    def __combine_current_words(self,currentWords,index1:int,index2:int) -> list:
        if(index1 < 0 or index2 < 0):
            raise ValueError("index 1 or 2 should be nonnegative numbers")
        twoGreater = index2 > index1
        word1 = currentWords[index1][0]
        word2 = currentWords[index2][0]
        if(self.__decrement_list(currentWords,index1) and index1 < index2):
            index2 -= 1
        self.__decrement_list(currentWords,index2)
        if(twoGreater):
            newWord = word2 + word1
        else:
            newWord = word1 + word2
        self.__add_to_list(currentWords,newWord)
        return currentWords
    def __decrement_list(self,currentList, index:int) -> bool:
        if(currentList[index][1] == 1):
            currentList.pop(index)
            return True
        else:
            currentList[index][1] -= 1
            return False
    def __add_to_list(self,currentWords,w:word) -> bool:
        for i in range(len(currentWords)):
            if(currentWords[i][0] ==w):
                currentWords[i][1] += 1
                return i
            if(currentWords[i][0] < w):
                currentWords.insert(i,[w,1])
                return i
        currentWords.append([w,1])
        return len(currentWords)-1
    def __next_smallest(self,index,currentList,excluded:set = set()):
        currentWord = currentList[index][0].string
        self.__decrement_list(currentList,index)
        letterWeight = np.zeros(self.n+1,dtype=int)
        removedLetter = currentWord[-1]
        letterWeight[removedLetter.rootIndex] = 1
        self.__add_to_list(currentList,word([removedLetter],letterWeight))
        currentWord = currentWord[:-1]
        newIndex = self.__add_to_list(currentList,word(currentWord,self.letter_list_to_weights(currentWord)))
        while True:
            flag = False
            for i in range(len(currentList),index):
                if(currentList[i][0] > removedLetter):
                    continue
                if(len(self.get_words(currentWords[-1][0].weights + currentWords[i][0].weights)) != 0
                    and currentList[i][0] not in excluded
                    ):
                        currentWords = self.__combine_current_words(currentWords,i,len(currentWords)-1)
                        flag = True
                        break
            if(not flag):
                self.__decrement_list(currentList,newIndex)
                remainingWeights = np.zeros(self.n+1,dtype=int)
                for i in currentList:
                    remainingWeights += currentList[i].weights * currentList[i][1]
                tempCurrentList = currentList
                self.__next_smallest(newIndex,currentList)
                break
        return currentList
    def SL_word_algo(self,weightsToGenerate) -> word:
        #if(sum(weightsToGenerate) > self.deltaHeight and self.isImaginary(sum(weightsToGenerate))):
        #    return []
        currentWords = []
        weightsToGenerate = np.array(weightsToGenerate,dtype=int)
        cofacPossible: bool = False
        #currentWeight = np.zeros(self.n + 1, dtype=int)
        for i in range(len(self.ordering)-1,-1,-1):
            if(weightsToGenerate[self.ordering[i].rootIndex] > 0): 
                arr = np.zeros(self.n+1,dtype=int)
                arr[self.ordering[i].rootIndex] = 1 
                currentWords.append([word([self.ordering[i]],arr),weightsToGenerate[self.ordering[i].rootIndex]])
            #if(len(currentWords) and self.isImaginary(sum(currentWords[0][0].weights))):
            #    self.__nextSmallest(0,currentWords)
        foundFlag = True
        while(foundFlag and len(currentWords) > 1):
            foundFlag = False
            if(currentWords[-1][1] == 1):
                pass
            for possibleAppendInd in range(len(currentWords)):
                if(possibleAppendInd == len(currentWords)-1 and currentWords[-1][1] == 1):
                    break
                leftWeight = currentWords[-1][0].weights
                rightCofac = currentWords[-1][0][word.list_cost_fac(currentWords[-1][0]):]
                if(self.contains_weight(leftWeight + currentWords[possibleAppendInd][0].weights)
                    and (not self.is_imaginary_height(currentWords[-1][0].height) or self.e_bracket(currentWords[-1][0] + currentWords[possibleAppendInd][0]))):
                        if(possibleAppendInd == len(currentWords)-1 
                           or (possibleAppendInd == len(currentWords)-2 and currentWords[-1][1]==1)):
                            cofacPossible = True
                        currentWords = self.__combine_current_words(currentWords,possibleAppendInd,len(currentWords)-1)
                        foundFlag = True
                        break
                if(cofacPossible and  
                   self.contains_weight(weightsToGenerate + self.letter_list_to_weights(rightCofac) - currentWords[-1][0].weights)):
                    potentialRightCofac = self.SL_word_algo(weightsToGenerate+self.letter_list_to_weights(rightCofac) - currentWords[-1][0].weights)
                    if(word.letter_list_cmp(potentialRightCofac,np.concatenate((rightCofac,currentWords[possibleAppendInd][0].string))) >= 0):
                        return word(np.concatenate((currentWords[-1][0].string[:-len(rightCofac)],potentialRightCofac.string)),weightsToGenerate)
        return currentWords[-1][0]
    def SL_word_rightFac_pred(self,weightsToGenerate) -> word:
                #if(sum(weightsToGenerate) > self.deltaHeight and self.isImaginary(sum(weightsToGenerate))):
        #    return []
        currentWords = []
        weightsToGenerate = np.array(weightsToGenerate,dtype=int)
        cofacPossible: bool = False
        #currentWeight = np.zeros(self.n + 1, dtype=int)
        for i in range(len(self.ordering)-1,-1,-1):
            if(weightsToGenerate[self.ordering[i].rootIndex] > 0): 
                arr = np.zeros(self.n+1,dtype=int)
                arr[self.ordering[i].rootIndex] = 1 
                currentWords.append([word([self.ordering[i]],arr),weightsToGenerate[self.ordering[i].rootIndex]])
            #if(len(currentWords) and self.isImaginary(sum(currentWords[0][0].weights))):
            #    self.__nextSmallest(0,currentWords)
        foundFlag = True
        while(foundFlag and len(currentWords) > 1):
            foundFlag = False
            if(currentWords[-1][1] == 1 and word.strict_letter_list_cmp(currentWords[-1][0].string,currentWords[-2][0].string)):
                return currentWords[-2][0]
            for possibleAppendInd in range(len(currentWords)):
                if(possibleAppendInd == len(currentWords)-1 and currentWords[-1][1] == 1):
                    break
                leftWeight = currentWords[-1][0].weights
                rightCofac = currentWords[-1][0][word.list_cost_fac(currentWords[-1][0]):]
                if(self.contains_weight(leftWeight + currentWords[possibleAppendInd][0].weights)
                    and (not self.is_imaginary_height(currentWords[-1][0].height) or self.e_bracket(currentWords[-1][0] + currentWords[possibleAppendInd][0]))):
                        if(possibleAppendInd == len(currentWords)-1 
                           or (possibleAppendInd == len(currentWords)-2 and currentWords[-1][1]==1)):
                            cofacPossible = True
                        currentWords = self.__combine_current_words(currentWords,possibleAppendInd,len(currentWords)-1)
                        foundFlag = True
                        break
                if(cofacPossible and  
                   self.contains_weight(weightsToGenerate + self.letter_list_to_weights(rightCofac) - currentWords[-1][0].weights)):
                    potentialRightCofac = self.SL_word_algo(weightsToGenerate+self.letter_list_to_weights(rightCofac) - currentWords[-1][0].weights)
                    if(word.letter_list_cmp(potentialRightCofac,np.concatenate((rightCofac,currentWords[possibleAppendInd][0].string))) >= 0):
                        return None
        #return currentWords[-1][0]
    def SL_word_algo_2(self,weightsToGenerate):
        weightsToGenerate = np.array(weightsToGenerate,dtype=int)
        currentWords = []
        for i in range(len(self.ordering)-1,-1,-1):
            if(weightsToGenerate[self.ordering[i].rootIndex] > 0): 
                arr = np.zeros(self.n+1,dtype=int)
                arr[self.ordering[i].rootIndex] = 1 
                currentWords.append([word([self.ordering[i]],arr),weightsToGenerate[self.ordering[i].rootIndex]])
        if(len(currentWords) == 1):
            return currentWords[0][0]
        while True:
            for i in range(currentWords[-1][1]):
                for possibleAppendInd in range(len(currentWords)):
                    if(len(self.get_words(currentWords[-1][0].weights + currentWords[possibleAppendInd][0].weights)) != 0):
                        if(possibleAppendInd == len(currentWords) -1 and currentWords[-1][1] == 1):
                            continue
                        currentWords = self.__combine_current_words(currentWords,possibleAppendInd,len(currentWords)-1)
                        break
            if(len(currentWords) == 1):
                return currentWords[0][0]
            if currentWords[-1][1] == 1:
                rightWeight = currentWords[-2][0].weights
                remainingWeight = weightsToGenerate - currentWords[-2][0].weights
                if(self.contains_weight(remainingWeight)):
                    return self.SL_word_algo_2(remainingWeight) + self.SL_word_algo_2(currentWords[-2][0].weights)
                k = 0
                while np.all((k+1)*self.delta <= remainingWeight):
                    k += 1
                maxWeight = None
                for w in self.baseWeights:
                    adjW = w+k*self.delta
                    comp = weightsToGenerate - adjW
                    if(np.any(comp < rightWeight) or np.any(adjW < currentWords[-1][0].weights) or 
                       not self.contains_weight(comp)):
                        continue
                    maxWeight = adjW
                return self.SL_word_algo_2(maxWeight) + self.SL_word_algo_2(weightsToGenerate - maxWeight)
if(__name__ == "__main__"):
    F4 = rootSystem([0,2,1,3,4],"F",1)
    F4.SL_word_algo([0, 2, 3, 4, 2])