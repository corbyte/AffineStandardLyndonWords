import numpy as np
import argparse
from scipy import sparse
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
    def __init__(self, wordArray,l,matrix=(0,0)):
        self.string = np.array(wordArray,dtype=letter)
        self.matrix = matrix
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
        return (np.add(np.matmul(A.matrix[0], B.matrix[0]) , -np.matmul(B.matrix[0], A.matrix[0])), B.matrix[1] + A.matrix[1])
    def __init__(self, ordering:letterOrdering):
        self.arr.append([word([i],len(ordering.letterOrdering)) for i in ordering.letterOrdering])
        self.ordering = ordering
    def getWord(self, combination):
        sameLengths = self.arr[sum(combination)-1]
        for i in sameLengths:
            if(np.all(i.weights == combination)):
                return i
        return None
    def genWord(self, combinations,affine=False,topn=1):
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
                if(other < j):
                    newWord = other + j
                    if(topn!=1):
                        bracket = standardLyndonWords.commuatator(other,j)
                        #Checks to see if bracket is non-zero
                        if not bracket.any():
                            continue
                        newWord.matrix = bracket
                    potentialOptions.append(newWord)
                else:
                    newWord = j + other
                    if(topn!=1):
                        bracket = standardLyndonWords.commutator(other,j)
                        if not bracket.any():
                            continue
                        newWord.matrix = bracket
                    potentialOptions.append(newWord)
        if(len(self.arr) < weight):
            self.arr.append([])
        if topn == 1:
            match = potentialOptions[0]
            for i in potentialOptions:
                if i > match:
                    match = i
            self.arr[weight-1].append(match)
            return match
        else:
            potentialOptions.sort()
            newWords = np.empty(topn,dtype=word)
            for i in range(topn):
                newWords[i] = potentialOptions[i]
            self.arr[weight-1].extend(newWords)
            return newWords
def parseWord(s:str):
    if(len(s) == 1):
        return s
    rightFactor = s[-1]
    for i in range(len(s)-1,0,-1):
        if(s[i]<rightFactor[0]):
            return
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type",choices=["C","c","A","a"])
    parser.add_argument("size",type=int)
    parser.add_argument("-o","--order", nargs='+', type =int)
    parser.add_argument('-a','--affine_count',type=int, default=0)
    #args = parser.parse_args()
    args = parser.parse_args(['A','2','-a','2'])
    type = args.type
    affineCount = args.affine_count
    size = args.size
    orderInput = args.order
    # parameter input: type affinecount size stanard order y/n order if n
    order = []
    if(orderInput is None):
        if(affineCount == 0):
            order = [letter(str(i),i) for i in range(1,size+1)]
        else:
            order = [letter(str(i),i) for i in range(1,size)]
            order.append(letter("0",0))
    else:
        order = [letter(str(i),i) for i in orderInput]
    ordering = letterOrdering(order)
    match type:
        case "A"|"a":
            Atype(size,ordering,affineCount)
        case "C"|"c":
            Ctype(size,ordering,affineCount)

def Ctype(size,ordering,affineCount=0,printIt=False):
    return genCTypeFinite(size,ordering,printIt=True)
def Atype(size,ordering,affineCount=0):
    if(affineCount == 0):
        sLyndonWords = genATypeFinite(size,ordering,printIt=True)
    else:
        sLyndonWords = genATypeAffine(size,ordering,affineCount,True)
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
    for i in range(len(sLyndonWords.arr[0])-1):
        arr = np.zeros((size,size),dtype=int)
        arr[i,i+1] = 1
        sLyndonWords.arr[0][i].matrix = (sparse.csr_array(arr),0)
    sLyndonWords.arr[0][-1].matrix = (sparse.csc_matrix(np.eye(size,dtype=int)),1)
    delta = np.ones(size,dtype = int)
    for deltaCount in range(affineCount+1):
        for length in range(1,size+1):
            if(length == 1 and deltaCount == 0):
                continue
            for start in range(0,size):
                if(length == size and start ==1):
                    break
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
                if(length == size):
                    if printIt:
                        print(sLyndonWords.genWord(comb,affine=True,topn=size-1))
                    else:
                        sLyndonWords.genWord(comb,affine=True,topn=size-1)
                else:
                    if printIt:
                        print(sLyndonWords.genWord(comb,affine=True))
                    else:
                        sLyndonWords.genWord(comb,affine=True)
    return sLyndonWords
if __name__ == '__main__':    
    main()