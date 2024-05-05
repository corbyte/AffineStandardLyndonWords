import numpy as np
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
        return self.index == other.index
    def __ne__(self,other):
        return not (self.index == other.index)
    def __geq__(self, other):
        return not (self < other)
    def __le__(self, other):
        return self.index <= other.index
    def __ge__(self,other):
        return self.index >= other.index
class word:
    #Maybe change to sparse matrix
    def __init__(self, wordArray,l,matrix=None):
        self.string = np.array(wordArray,dtype=letter)
        self.matrix = None
        self.weights = np.zeros(l,dtype=int)
        for i in self.string:
            self.weights[i.rootIndex-1] += 1
    def __str__(self):
        return ','.join(str(i) for i in self.string)
    def __eq__(self,other):
        if(len(self.string) != len(other.string)):
            return False
        for i in range(len(self.string)):
            if(self.string[i] != other.string[i]):
                return False
        return True
    def __lt__(self,other):
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
    def __ne__(self,other):
        return not (self == other)
    def __add__(self,other):
        return word(np.concatenate((self.string,other.string),dtype=word),len(self.weights)) 
class letterOrdering:
    letterOrdering:list[letter]
    def __init__(self, letterOrdering):
        self.letterOrdering = letterOrdering
        for i in range(len(letterOrdering)):
            self.letterOrdering[i].index = i
class standardLyndonWords:
    arr = []
    ordering:letterOrdering
    def commutator(A,B):
        return np.add(np.matmul(A, B) , np.matmul(B, A))
    def __init__(self, ordering):
        self.arr.append([word([i],len(ordering.letterOrdering)) for i in ordering.letterOrdering])
        self.ordering = ordering
    def getWord(self, combination):
        sameLengths = self.arr[sum(combination)-1]
        for i in sameLengths:
            if(np.all(i.weights == combination)):
                return i
        return None
    def genWord(self, combinations,affine=False):
        weight = sum(combinations)
        potentialOptions = []
        if (weight > len(self.arr)+1):
            return None
        for i in range(1,weight//2 + 1):
            minlen = self.arr[i-1]
            for j in minlen:
                diff = list(map(lambda a, b: a-b, combinations,j.weights))
                if(min(diff) >= 0):
                    other = self.getWord(diff)
                    if other is None:
                        continue
                else:
                    continue
                if( other < j):
                    potentialOptions.append(other + j)
                else:
                    potentialOptions.append(j + other)
        if(len(self.arr) < weight):
            self.arr.append([])
        match = potentialOptions[0]
        for i in potentialOptions:
            if match < i:
                match = i
        self.arr[weight-1].append(match)
        return match
def parseWord(s:str):
    if(len(s) == 1):
        return s
    rightFactor = s[-1]
    for i in range(len(s)-1,0,-1):
        if(s[i]<rightFactor[0]):
            return
def main():
    print("Select type:")
    print("A")
    print("C")
    type = input()
    affineCount=0
    print("Affine?(y/n)")
    match input():
        case "y"|"Y":
            print("Enter Affine Count:")
            affineCount = int(input())
    print("Enter size:")
    size = int(input())
    print("Standard order [y/n]:")
    so = input()
    ordered = np.empty(size,dtype=letter)
    match so:
        case "y" | "Y":
            if(affineCount != 0):
                ordered = [letter(str(i),i) for i in range(1,size)]
                ordered.append(letter("0",0))
            else:
                ordered = [letter(str(i),i) for i in range(1,size+1)]
        case _:
            for i in range(1,size+1):
                print(f"Enter root index for {i} ordered element")
                index = input()
                ordered[i-1]= letter(index,int(index))
    ordering = letterOrdering(ordered)
    match type:
        case "A"|"a":
            Atype(size,ordering,affineCount)
        case "C"|"c":
            Ctype(size,ordering,affineCount)

def Ctype(size,ordering,affineCount=0,printIt=False):
    return genCTypeFinite(size,ordering,printIt=True)
def Atype(size,ordering,affineCount=0):
    sLyndonWords = genATypeFinite(size,ordering,printIt=True)
    return sLyndonWords
def genCTypeFinite(size:int,ordering:letterOrdering,printIt=False):
    sLyndonWords = standardLyndonWords(ordering)
    for length in range(2,2*size):
        if(length <= 2*size -2):
            #i+j
            comb = np.zeros(size,dtype=int)
            for i in range(size-min(length,size),size):
                comb[i] = 1
            oneIndex = size-min(length,size)
            twoIndex = size-1
            while(sum(comb) < length):
                twoIndex -= 1
                comb[twoIndex] = 2
            while(oneIndex < twoIndex):
                if(printIt):
                    print(sLyndonWords.genWord(comb))
                else:
                    sLyndonWords.genWord(comb)
                twoIndex -= 1
                comb[twoIndex] = 2
                comb[oneIndex] = 0
                oneIndex+=1
        #i-j
        if(length < size):
            for start in range(0,size-length):
                comb = np.zeros(size,dtype=int)
                for k in range(start,start+length):
                    comb[k] =1
                if printIt:
                    print(sLyndonWords.genWord(comb))
                else:
                    sLyndonWords.genWord(comb)
        #2i
        if(length % 2 == 1):
            comb=np.zeros(size,dtype=int)
            comb[-1] = 1
            for k in range(size-2,size-2-length//2,-1):
                comb[k] = 2
            if printIt:
                print(sLyndonWords.genWord(comb))
            else:
                sLyndonWords.genWord(comb)
    return sLyndonWords
def genATypeFinite(size:int,ordering:letterOrdering,printIt=False):
    sLyndonWords = standardLyndonWords(ordering)
    for length in range(2,size+1):
        for start in range(0,size - length + 1):
            comb = np.zeros(size,dtype=int)
            for k in range(start,start+length):
                comb[k] = 1
            if printIt:
                print(sLyndonWords.genWord(comb))
            else:
                sLyndonWords.genWord(comb)
    return sLyndonWords
def genATypeAffine(size:int,ordering:letterOrdering,affineCount,printIt=False):
    sLyndonWords = standardLyndonWords(ordering)
    delta = np.ones(size,dtype=int)
    for deltaCount in range(affineCount+1):
        for length in range(1,size):
            if(length == 1 and deltaCount == 0):
                continue
            for start in range(0,size):
                if(start + length < size):
                    comb = np.zeros(size,dtype=int)
                    for k in range(length):
                        comb[start+k] = 1
                else:
                    comb=np.ones(size,dtype=int)
                    startprime = start+length -size
                    for k in range(size-length):
                        comb[startprime+k] = 0
                comb = comb + deltaCount*delta
            if printIt:
                print(sLyndonWords.genWord(comb,affine=True))
            else:
                sLyndonWords.genWord(comb,affine=True)
    return sLyndonWords
if __name__ == '__main__':
    main()