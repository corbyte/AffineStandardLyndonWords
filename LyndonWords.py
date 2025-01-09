import numpy as np
class letter:
    """Class for letters of words"""
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
    """Class for words, as a string of letters"""
    def __init__(self, wordArray,weights):
        """Initialization of word object
        
        wordArray -- iterable of letters
        weights -- number of occurances of each letter in the wordArray, given as an iterable
        """
        self.string = np.array(wordArray,dtype=letter)
        self.hs = None
        self.weights = np.array(weights)
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
        """comparision method for two words
        
        If result is 
            2, then first > second and second is not a substring of first
            1, then first > second and second is a substring of first
            0, then first is equal to second
            -1, then first < second and first is a substring of second
            -2, then first < second and first is not a substring of second
        """
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
                return 2
            if(first[i].index < second[i].index):
                return -2
        if(lFirst < lSecond):
            return -1
        if(lFirst > lSecond):
            return 1
        return 0
    def list_cost_fac(list):
        """Cofactorization for a string of letters instead of a word"""
        smallestIndex = len(list)-1
        for i in range(len(list)-1,0,-1):
            if(word.letter_list_cmp(list[smallestIndex:],list[i:]) > 0):
                smallestIndex = i
        return smallestIndex        
    def no_commas(self):
        """returns the word without commas as a string"""
        return ''.join(str(i) for i in self.string)
    def lca(first,second):
        if(len(first) < len(second)):
            minlen = len(first)
        else:
            minlen = len(second)
        for i in range(minlen):
            if(first[i] != second[i]):
                if(i == 0):
                    return []
                return first[:i]
        return first[:minlen]
class letterOrdering:
    """Class used to contain the ordering of letters in a RootSystem"""
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
        """Returns mapping of letters to their order
        
        e.g. If order is 2<1<3<0 -> [3,1,0,2]"""
        lst = np.zeros(len(self.order),dtype=object)
        for let in self.order:
            lst[let.rootIndex] = let
        return lst
    def to_ordered_list(self) -> np.ndarray:
        """Returns letter in order
        
        e.g. 2<1<3<0 -> [2,1,3,0]"""
        return np.array(self.order,dtype=object)
class rootSystem:
    """Class to represent a rootsystem"""
    def e_bracket(self, bracketWord:word) -> bool:
        """Determines if the bracketing of a word is non-zero, and the word is real
        
        returns True if the bracketing is non-zero"""
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
    def split_e_bracket(self,hs,realword) -> bool:
        weights = realword.weights - (self.delta * realword.weights[0])
        return np.dot(hs ,(self.sym_matrix @ weights[1:])) != 0
    def h_bracket(self,word:word) -> np.array:
        """Determines the bracketing of an imaginary word
        
        Returns a vector representing the weights of the h_i
        
        It will be of length n"""
        a = self.costfac(word)[0]
        if(a is None):
            return np.zeros(self.n,dtype=int)
        word.cofactorizationSplit = a.height
        newA = (a.weights - (self.delta *a.weights[0]))
        if(np.any(newA < 0)):
            return -newA[1:]
        return newA[1:]
    def list_h_bracketing(self,letterList) -> bool:
        """Same as h_bracket but for list of letters"""
        letterlist = letterList[:word.list_cost_fac(letterList)]
        weights = self.letter_list_to_weights(letterlist)
        newA = (weights - (self.delta *weights[0]))
        if(np.any(newA < 0)):
            return -newA[1:]
        return newA[1:]
    def list_e_bracketing(self,letterList):
        """Same as e_bracket but for list of letters"""
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
        """Initializationof root system
        
        ordering -- list of ordering for the rootsystem
        type -- type of the rootsystem
        """
        self.type = type.upper()
        if( len(self.type) != 1 or self.type < 'A' or self.type > 'G' ):
            raise ValueError('Type is invalid')
        self.n = len(ordering)-1
        self.ordering:letterOrdering = letterOrdering(ordering)
        self.weightToWordDictionary:dict = {}
        self.minWord:word = None
        def weightsGeneration(letter):
            #Generates simple root vectors
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
        self.baseWeights = rootSystem.get_base_weights(self.type,self.n)
        self.numberOfBaseWeights = len(self.baseWeights)
        self.cartan_matrix = rootSystem.get_cartan_matrix(self.type,self.n)
        self.delta = rootSystem.get_delta(self.type,self.n)
        self.delta.flags.writeable = False
        self.deltaHeight = sum(self.delta)
        self.vectors_norm2 = rootSystem.basis_vector_norm2(self.type,self.n)
        self.sym_matrix = self.get_sym_matrix()
        self._maxWord:word  = self.get_words(weightsGeneration(self.ordering[-1]))[0]
    def get_base_weights(type,n):
        """Returns roots of height <= delta for a certain type and n"""
        if(type == 'A'):
            return rootSystem.A_weights(n)
        elif(type == 'B'):
            return rootSystem.B_weights(n)
        elif(type == 'C'):
            return rootSystem.C_weights(n)
        elif(type =='D'):
            return rootSystem.D_weights(n)
        elif(type == 'E'):
            return rootSystem.E_weights(n)
        elif(type == 'F'):
            return rootSystem.F_weights()
        elif(type == 'G'):
            return rootSystem.G_weights()
    def get_delta(type,n) -> np.array:
        """Returns delta for a certain type and n"""
        if(type == 'A'):
            return rootSystem.A_delta(n)
        elif (type == 'B'):
            return rootSystem.B_delta(n) 
        elif(type == 'C'):
            return rootSystem.C_delta(n)
        elif(type == 'D'):
            return rootSystem.D_delta(n)
        elif(type == 'E'):
            return rootSystem.E_delta(n)
        elif(type == 'F'):
            return rootSystem.F_delta()
        elif(type == 'G'):
            return rootSystem.G_delta()
        raise ValueError("Incorrect value for type")
    def A_weights(n) -> np.array:
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
    def B_weights(n) -> np.array:
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
    def C_weights(n) -> np.array:
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
    def D_weights(n) -> np.array:
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
        return np.array(arr,dtype=int)
    def E_weights(n,affine:bool=True) ->np.array:
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
    def F_weights(affine:bool=True) -> np.array:
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
        """Generates the affine roots from the simple lie algebra
        
        Used in the get weights function"""
        for i in arr:
            if(i[0] == 1):
                break
            arr.append(delta - i)
        arr.append(delta)
    def costfac(self,wordToFactor:word):
        """Returns the costandard factorization of a word as a tuple of 2 words"""
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
    def standfac(self,wordToFactor:word):
        """Returns the standard factorization of a word as a tuple of 2 words"""
        if(wordToFactor.height == 1):
            return (wordToFactor,None)
        weight = np.copy(wordToFactor.weights)
        for i in range(wordToFactor.height-1,0,-1):
            weight[wordToFactor.string[i].rootIndex] -= 1
            leftWords = self.__get_words(weight)
            for lWord in leftWords:
                if(word.letter_list_cmp(lWord.string,wordToFactor.string[:i]) == 0):
                    for rWord in self.__get_words(wordToFactor.weights - weight):
                        if(word.letter_list_cmp(rWord.string,wordToFactor.string[i:]) == 0):
                            return (lWord,rWord)
        return (None,None)
    def __get_words(self, combination:np.array):
        """Gets all words corresponding to a certain root"""
        return self.weightToWordDictionary.get(combination.tobytes(),[])
    def get_words(self, combination):
        """Gets all words corresponding to a certain root"""
        if(self.contains_weight(combination)):
            ret = self.__get_words(np.array(combination, dtype=int))
            if(len(ret) == 0):
                self.generate_up_to_height(sum(combination))
                ret = self.__get_words(np.array(combination, dtype=int))
            return ret
        return []
    def get_chain(self,weight,min_delta = 2):
        """Gets the string of word weight, weight+\delta \cdots for all generated words"""
        self.generate_up_to_delta(min_delta)
        matches = []
        k=0
        newWord = self.__get_words(weight + k*self.delta)
        while len(newWord) > 0:
            matches.extend(newWord)
            k+=1
            newWord=self.__get_words(weight + k*self.delta)
        return matches
    def is_imaginary_height(self,height:int):
        """Checks if a word has imaginary height"""
        return height % self.deltaHeight == 0
    def __gen_word(self, combinations:np.array):
        """Function called to generate a new word"""
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
        """Gets words grouped together by base word
        
        Used primarily for csv output"""
        returnarr = []
        for i in self.baseWeights:
            returnarr.append(np.array(self.get_chain(i)))
        return np.array(returnarr)
    def get_monotonicity(self, weights,deltaIndex = 0):
        """Gets monotonicity of a chain of words
        
        < 0 decreasing
        = 0 not monotone
        > 0 increasing
        """
        self.generate_up_to_delta(2)
        words = self.get_chain(weights)
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
        """Filter for checking monotonicity of all words"""
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
    def check_convexity(self,deltagen=2,word_convexity=False):
        """Checks convexity on the rootsystem"""
        self.generate_up_to_delta(deltagen)
        for w in list(self.weightToWordDictionary.values()):
            if(len(w) > 1):
                continue
            currentWord = w[0]
            for decomp in self.get_decompositions(currentWord.weights):
                if(self.is_imaginary_height(sum(decomp[0])) or self.is_imaginary_height(sum(decomp[1]))):
                    if(not word_convexity):
                        continue
                    leftim = False
                    if(self.is_imaginary_height(sum(decomp[0]))):
                        imheight = decomp[0]
                        reword = self.get_words(decomp[1])[0]
                        leftim = True
                    else:
                        imheight = decomp[1]
                        reword = self.get_words(decomp[0])[0]
                    max_found = False
                    for i in self.get_words(imheight):
                        if(not max_found and self.split_e_bracket(i.hs,reword)):
                            if(leftim):
                                leftWord = i
                                rightWord = reword
                                max_found = True
                            else:
                                rightWord = i
                                leftWord = reword
                                max_found = True
                        if(max_found and 
                           ((i < currentWord) != (i < reword))):
                            yield (currentWord,self.get_words(decomp[0])[0],self.get_words(decomp[1])[0])
                            break
                else:
                    leftWord = self.get_words(decomp[0])[0]
                    rightWord = self.get_words(decomp[1])[0]
                if(leftWord > rightWord):
                    continue
                if(leftWord > currentWord):
                    yield (currentWord,self.get_words(decomp[0])[0],self.get_words(decomp[1])[0])
                if(rightWord < currentWord):
                    yield (currentWord,self.get_words(decomp[0])[0],self.get_words(decomp[1])[0])
    def __get_decompositions(self,weights):
        """Gets all decompositions of a word"""
        if(weights is not np.array):
            weights = np.array(weights)
        returnarr = []
        delta = 0
        while(delta*self.deltaHeight< sum(weights)):
            for i in self.baseWeights:
                if len(self.__get_words(weights-i - delta*self.delta)) > 0:
                    returnarr.append((i + delta*self.delta,weights-i-delta*self.delta))
            delta+= 1 
        return returnarr
    def get_decompositions(self,weightToDecompose):
        """Gets all decompositions of a word"""
        if(weightToDecompose is not np.array):
            weightToDecompose = np.array(weightToDecompose)
        deltaSum = self.deltaHeight
        deltaIndex = 0
        while(deltaIndex*deltaSum < sum(weightToDecompose)):
            for i in self.baseWeights:
                if(sum(i) + deltaSum*deltaIndex >= sum(weightToDecompose)):
                    continue
                potential_weight = weightToDecompose - (i + deltaIndex * self.delta)
                if(np.any(potential_weight < 0)):
                    continue
                potential_weight = potential_weight - self.delta*((sum(potential_weight)-1)//deltaSum)
                for bw in self.baseWeights:
                    if(np.all(potential_weight == bw)):
                        yield (i+deltaIndex*self.delta,weightToDecompose-i-deltaIndex*self.delta)
            deltaIndex+= 1
    def general_critical_roots(self,degree):
        for i in self.get_decompositions(degree):
            left = i[0]
            right = i[1]
            rightprime = np.copy(right)
            flag = True
            while((sum(rightprime)-1)//(sum(left)) >= 1):
                rightprime = rightprime - left
                if(not self.contains_weight(rightprime)):
                    flag = False
                    break
            if(not flag):
                continue
            oneOver = False
            continueflag = False
            for j in range(len(i[0])):
                if((rightprime - left)[j] > 1):
                    continueflag = True
                    break
                if((rightprime - left)[j] == 1):
                    if(oneOver):
                        continueflag = True
                        break
                    else:
                        oneOver = True
            if(continueflag):
                continue
            yield (left,right)
    def print_general_critical_roots(self,degree):
        for i in self.general_critical_roots(degree):
            print(i)
    def realized_critical_roots(self,degree):
        for left,right in self.general_critical_roots(degree):
            leftstrs = self.get_words(left)
            rightstrs = self.get_words(right)
            for leftstr in leftstrs:
                for rightstr in rightstrs:
                    temprightstr = rightstr.string
                    flag = False
                    while(len(temprightstr) > len(leftstr)):
                        if(word.letter_list_cmp(leftstr,temprightstr[:len(leftstr)]) == 0):
                            temprightstr = temprightstr[len(leftstr):]
                        else:
                            flag = True
                            break
                    if(flag):
                        continue
                    if((len(temprightstr) == 1 and word.letter_list_cmp(temprightstr,leftstr.string) > 0 )
                        or((word.letter_list_cmp(leftstr.string[:len(temprightstr)-1],temprightstr[:-1]) == 0)
                            and rightstr > leftstr)):
                            yield (leftstr,rightstr)
                                                        
    def print_realized_critical_roots(self,degree,printfunc = None):
        if(printfunc == None):
            printfunc = (lambda x: f"{x[0].no_commas()}:{x[1].no_commas()}")
        elif(printfunc == "delta_format"):
            printfunc = (lambda x: f"{self.parse_to_delta_format(x[0])}:{self.parse_to_delta_format(x[1])}")
        elif(printfunc == 'weights'):
            printfunc = (lambda x: f"{x[0].weights}:{x[1].weights}")
        elif(printfunc == 'mod_weight'):
            printfunc = (lambda x: f"{self.mod_delta(x[0].weights)}:{self.mod_delta(x[1].weights)}")
        for i in self.realized_critical_roots(degree):
            print(printfunc(i))
    def get_critical_roots(self,rootindex,k):
        minuspart = np.zeros(self.n+1,dtype=int)
        minuspart[rootindex] = 1
        decomps = self.get_decompositions(k*self.deltas)
        retlist = []
        for i in decomps:
            rightless = i[1] - minuspart
            if(self.is_imaginary_height(sum(i[0]))):
                continue
            if(min(rightless) < 0):
                continue
            if(min(i[0] - rightless) >= 0):
                retlist.append(i[0])
        return retlist
    def mod_delta(self,root):
        deltamult = sum(root)//(self.deltaHeight)
        modroot = root - deltamult*self.delta
        return (modroot,deltamult)
    def get_critical_pairs(self,rootindex,k):
        for left in self.get_critical_roots(rootindex,k):
            leftstr = self.get_words(left)[0]
            rightstr = self.get_words(k*self.delta - left)[0]
            if(len(rightstr) == 1
               or(word.letter_list_cmp(leftstr.string[:len(rightstr)-1],rightstr.string[:-1]) == 0
                  and rightstr > leftstr
                  and rightstr[-1].rootIndex == rootindex)):
                yield (leftstr,rightstr)
    def print_critical_pairs(self,rootindex,k):
        for (leftstr,rightstr) in self.get_critical_pairs(rootindex,k):
                print(f"{leftstr.no_commas()}:{rightstr.no_commas()}")
    def get_potential_words(self,weights):
        """This returns all other words that could be factors of a given word"""
        decomps = self.__get_decompositions(weights)
        arr =[]
        for i in decomps:
            for j in self.__get_words(i[0]):
                for k in self.__get_words(i[1]):
                    if(j < k):
                        arr.append(j+k)
                    else:
                        arr.append(k+j)
        return arr
    def A_delta(n:int) -> np.array:
        return np.ones(n+1,dtype=int)
    def B_delta(n:int) -> np.array:
        arr = np.repeat(2,n+1)
        arr[0] = 1
        arr[1] = 1
        return arr
    def C_delta(n:int) -> np.array:
        delta = np.repeat(2,n+1)
        delta[0] = 1
        delta[-1] = 1
        return delta
    def D_delta(n:int) -> np.array:
        delta = np.repeat(2,n+1)
        delta[0] =1
        delta[-1] = 1
        delta[-2] = 1
        delta[1] = 1
        return delta
    def E_delta(n:int) -> np.array:
        if(n == 6):
            return np.array([1,1,2,3,2,1,2],dtype=int)
        if(n == 7):
            return np.array([1,1,2,3,4,3,2,2],dtype=int)
        if(n == 8):
            return np.array([1,2,3,4,5,6,4,2,3])
        raise ValueError("n must be 6,7,8")
    def F_delta() -> np.array:
        return np.array([1,2,3,4,2],dtype=int)
    def G_delta() -> np.array:
        return np.array([1,2,3],dtype=int)
    def basis_vector_norm2(t:str, n:int) -> np.array:
        """Calculates the length squared of the root vectors"""
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
            for i in range(len(values)):
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
                values[0] = 3
                values[1] = 1
                return values
            else:
                raise ValueError("Invalid n for Type G")
    def get_sym_matrix(self):
        """Returns the symmetric cartan matrix"""
        diag = np.zeros((self.n,self.n),dtype=int)
        for i in range(self.n):
            diag[i,i] = self.vectors_norm2[i]
        return diag @ self.cartan_matrix
    def print_format(words, formatfunc):
        """prints words given a format"""
        for word in words:
            print(formatfunc(word))
    def parse_to_delta_format(self,parseWord:word):
        """returns word in terms of SL_i(\delta)
        
        The first number of the tuple represents i and the second,
        the number of times it occurs in a row
        """
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
        """returns the cartan matrix for a give type and n"""
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
                return np.array([[2,-1],[-3,2]], dtype=int)
        raise ValueError("Invalid parameters")
    def get_min_right_factor(self,root):
        minword = self._maxWord
        for l,r in self.get_decompositions(root):
            if(self.is_imaginary_height(sum(l)) or self.is_imaginary_height(sum(r))):
                if(self.is_imaginary_height(sum(l))):
                    imweight = l
                    realword = self.get_words(r)[0]
                else:
                    imweight = r
                    realword = self.get_words(r)[0]
                for imword in self.get_words(imweight):
                    if(self.split_e_bracket(imword.hs,realword)):
                        if(word.letter_list_cmp(imword.string,realword.string) > 0):
                            if(word.letter_list_cmp(imword.string,minword.string) < 0):
                                minword = imword
                        else:
                            if(word.letter_list_cmp(realword.string,minword.string) < 0):
                                minword = realword
                        break
            else:
                lword:word = self.get_words(l)[0]
                rword:word = self.get_words(r)[0]
                if(word.letter_list_cmp(lword.string,rword.string) >= 0):
                    continue
                if(word.letter_list_cmp(rword.string,minword.string) < 0):
                    minword = rword
        return minword
    def get_periodicity(self, simpleRoot,slIndex:int = 0,kdelta:int = 3) -> int:
        """Returns the periodicity of a simple root in a rootsystem
        
        slIndex -- used in imaginary words
        kdelta -- the min height to look at
        """
        #TODO: Updated based on slightly different definition of periodicity
        factors = []
        self.generate_up_to_delta(kdelta)
        for i in self.get_chain(simpleRoot):
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
        """Generate all words in the rootsystem upto a certain height"""
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
        """Generates all words in the rootsytsem upto a certain k\delta"""
        self.generate_up_to_height(self.deltaHeight*k)
    def contains_weight(self,weights):
        """Check if a root in is the rootsystem"""
        if(len(weights) != self.n+1):
            return False
        weights = np.array(weights,dtype=int)
        weights -= ((sum(weights)-1)//self.deltaHeight) * self.delta
        return np.any(np.all(self.baseWeights[:] == weights,axis=1))
    def letter_list_to_weights(self,letterList):
        """Returns a list of letters to vector form"""
        arr = np.zeros(self.n + 1,dtype=int)
        for l in letterList:
            arr[l.rootIndex] += 1
        return arr
    def get_max_im_word(self,weight,n=1):
        word = self.get_words(weight)[0]
        imwords = self.get_words(self.delta * n)
        for w in imwords:
            if(self.split_e_bracket(w.hs,word)):
                return w