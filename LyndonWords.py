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
    def __init__(self, wordArray,l,imaginary=False,matrix=(0,0)):
        self.string = np.array(wordArray,dtype=letter)
        self.imaginary=imaginary
        self.matrix = matrix
        self.weights = np.zeros(l,dtype=int)
        for i in self.string:
            self.weights[i.rootIndex-1] += 1
        self.weights.flags.writeable = False
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
class standardLyndonWords:
    ordering:letterOrdering
    def commutator(A,B):
        return (A.matrix[0]@ B.matrix[0]-B.matrix[0]@ A.matrix[0], B.matrix[1] + A.matrix[1])
    def __init__(self, ordering:letterOrdering):
        self.arr = []
        self.arr.append([word([i],len(ordering)) for i in ordering.order])
        self.ordering = ordering
    def getWords(self, combination):
        sameLengths = self.arr[sum(combination)-1]
        word = []
        for i in sameLengths:
            if(np.all(i.weights == combination)):
                word.append(i)
        return word
    def genWord(self, combinations,affine=False,topn=1):
        weight = sum(combinations)
        #Maybe make potentialOptions a set
        potentialOptions = []
        if (weight > len(self.arr)+1):
            return None
        for i in range(1,weight//2 + 1):
            minlen = self.arr[i-1]
            for j in minlen:
                diff = list(map(lambda a, b: a-b, combinations,j.weights))
                if(min(diff) < 0):
                    continue
                complement = self.getWords(diff)
                if len(complement) == 0:
                    continue
                for comp in complement:
                    if(comp < j):
                        newWord = comp + j
                        if(affine):
                            bracket = standardLyndonWords.commutator(comp,j)
                            #Checks to see if bracket is non-zero
                            if bracket[0].size == 0:
                                continue
                            newWord.matrix = bracket
                        potentialOptions.append(newWord)
                    else:
                        newWord = j + comp
                        if(affine):
                            bracket = standardLyndonWords.commutator(j,comp)
                            if bracket[0].size == 0:
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
            potentialOptions = list(set(potentialOptions))
            potentialOptions.sort(reverse=True)
            liPotentialOptions = [potentialOptions[0]]
            liset = [potentialOptions[0].matrix[0].toarray().flatten()]
            index = 1
            while(len(liset) < topn):
                #change to only use non-zero matrix entries
                liprime = liset + [potentialOptions[index].matrix[0].toarray().flatten()]
                if(np.linalg.matrix_rank(np.vstack(liprime)) == len(liprime)):
                    liset = liprime
                    liPotentialOptions.append(potentialOptions[index])
                index += 1
            self.arr[-1].extend(liPotentialOptions)
            return liPotentialOptions
def parseWord(s:str):
    if(len(s) == 1):
        return s
    rightFactor = s[-1]
    for i in range(len(s)-1,0,-1):
        if(s[i]<rightFactor[0]):
            return
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type",choices=["C","c","A","a","b","B",'d','D'])
    parser.add_argument("size",type=int)
    parser.add_argument("-o","--order", nargs='+', type =int)
    parser.add_argument('-a','--affine_count',type=int, default=0)
    args = parser.parse_args()
    #args = parser.parse_args(['C','2', '-a' ,'2', '-o', '2' ,'1','0'])
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
            match type:
                case 'A'|'a':
                    order = [int(i) for i in range(1,size+1)]
                    order.append(0)
                case 'C'|'c':
                    order = list(size+1)
                    order[0]
                    for i in range(0,size):
                        order[i+1] = i
                case _:
                    order = [int(i) for i in range(1,size+1)]
                    order.append(0)
    else:
        order = [int(i) for i in orderInput]
    ordering = letterOrdering(order)
    match type:
        case "A"|"a":
            Atype(ordering,affineCount)
        case "B"|"b":
            Btype(ordering,affineCount)
        case "C"|"c":
            Ctype(ordering,affineCount)
        case "D"|"d":
            Dtype(ordering,affineCount)

def Btype(ordering,affineCount=0):
    return genBTypeFinite(ordering,True)
def Ctype(ordering,affineCount=0):
    if(affineCount == 0):
        sLyndonWords = genCTypeFinite(ordering,True)
    else:
        sLyndonWords = genCTypeAffine(ordering,affineCount,True)
    return sLyndonWords
def Dtype(ordering,affineCount=0):
    return genDTypeFinite(ordering,printIt=True)
def Atype(ordering,affineCount=0):
    if(affineCount == 0):
        sLyndonWords = genATypeFinite(ordering,printIt=True)
    else:
        sLyndonWords = genATypeAffine(ordering,affineCount,True)
    return sLyndonWords
def genBTypeFinite(ordering:letterOrdering,printIt=False):
    sLyndonWords = standardLyndonWords(ordering)
    size = len(ordering)
    for length in range(2,2*size):
        #i
        if(length <= size):
            comb = np.zeros(size,dtype=int)
            for i in range(size-length,size):
                comb[i] = 1
            if printIt:
                print(sLyndonWords.genWord(comb))
            else:
                sLyndonWords.genWord(comb)
            if(length != size):
                #ei - ej
                for start in range(0,size - length):
                    comb = np.zeros(size,dtype=int)
                    for k in range(start,start+length):
                        comb[k] = 1
                    if printIt:
                        print(sLyndonWords.genWord(comb))
                    else:
                        sLyndonWords.genWord(comb)
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
                if(printIt):
                    print(sLyndonWords.genWord(comb))
                else:
                    sLyndonWords.genWord(comb)
                twoIndex -= 1
                comb[twoIndex] = 2
                comb[oneIndex] = 0
                oneIndex+=1
    return sLyndonWords
        
def genCTypeFinite(ordering:letterOrdering,printIt=False):
    size=len(ordering)
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
def genCTypeAffine(ordering:letterOrdering,affineCount,printIt=False):
    size=len(ordering)
    sLyndonWords = standardLyndonWords(ordering)
    for i in range(len(sLyndonWords.arr[0])):
        rootIndex = sLyndonWords.arr[0][i][0].rootIndex
        matrix = np.zeros((2*(size-1),2*(size-1)),dtype=int)
        if(rootIndex == 0):
            matrix[-1,0] = 1
            t=1
        elif rootIndex == size-1:
            matrix[size-2,size-1] = 1
            t=0
        else:
            matrix[rootIndex-1,rootIndex] = 1
            matrix[-rootIndex-1,-(rootIndex)] = -1
            t=0
        sLyndonWords.arr[0][i].matrix = (sparse.csr_array(matrix),t)
    delta = np.repeat(2,size)
    delta[-1] = 1
    delta[-2] = 1
    simpleLetterOrdering = genCTypeFinite(letterOrdering([i for i in range(1,size)])).arr
    weights = [None]*len(simpleLetterOrdering)
    for row in range(len(simpleLetterOrdering)):
        weights[row] = [None]*len(simpleLetterOrdering[row])
        for root in range(len(simpleLetterOrdering[row])):
            weights[row][root] = simpleLetterOrdering[row][root].weights.tolist()+ [0]
    for deltaCount in range(affineCount+1):
        if(deltaCount > 0):
            if printIt:
                print(*sLyndonWords.genWord(deltaCount*delta,True,size-1),sep='\n')
            else:
                sLyndonWords.genWord(deltaCount*delta,True,size-1)
        for length in range(1,2*size-2):
            if(deltaCount > 0 or length > 1):
                for comb in weights[length-1]:
                    if printIt:
                        print(sLyndonWords.genWord(comb+deltaCount*delta,True))
                    else:
                        sLyndonWords.genWord(comb+deltaCount*delta,True)
                for comb in weights[-length]:
                    if printIt:
                        print(sLyndonWords.genWord((deltaCount+1)*delta - comb,True))
                    else:
                        sLyndonWords.genWord((deltaCount+1)*delta - comb,True)
            
        
def genATypeFinite(ordering:letterOrdering,printIt=False):
    size = len(ordering)
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
def genATypeAffine(ordering:letterOrdering,affineCount,printIt=False):
    size = len(ordering)
    sLyndonWords = standardLyndonWords(ordering)
    for i in range(len(sLyndonWords.arr[0])):
        rootIndex = sLyndonWords.arr[0][i][0].rootIndex
        if(rootIndex == 0):
            arr = np.zeros((size,size),dtype=int)
            arr[-1,0] = -1
            t=1
        else:
            arr = np.zeros((size,size),dtype=int)
            arr[rootIndex-1,rootIndex] = 1
            t=0
        sLyndonWords.arr[0][i].matrix = (sparse.csr_array(arr),t)
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
                if(length == size and size != 2):
                    if printIt:
                        for i in sLyndonWords.genWord(comb,affine=True,topn=size-1):
                            print(f"{i} {comb}")
                    else:
                        sLyndonWords.genWord(comb,affine=True,topn=size-1)
                else:
                    if printIt:
                        print(f"{sLyndonWords.genWord(comb,affine=True)} {comb}")
                    else:
                        sLyndonWords.genWord(comb,affine=True)
    return sLyndonWords
def genDTypeFinite(ordering:letterOrdering,printIt=False):
    size=len(ordering)
    sLyndonWords = standardLyndonWords(ordering)
    for length in range(2,2*size-2):
        #i-j
        for start in range(0,size - length):
            comb = np.zeros(size,dtype=int)
            for k in range(start,start+length):
                comb[k] = 1
            if printIt:
                print(sLyndonWords.genWord(comb))
            else:
                sLyndonWords.genWord(comb)
        #i+j
        if(length < size):
            comb=np.zeros(size,dtype=int)
            comb[-1] =1
            for i in range(-(length+1),-2,1):
                comb[i] = 1
            if printIt:
                print(sLyndonWords.genWord(comb))
            else:
                sLyndonWords.genWord(comb)
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
                if(printIt):
                    print(sLyndonWords.genWord(comb))
                else:
                    sLyndonWords.genWord(comb)
                twoIndex -= 1
                comb[twoIndex] = 2
                comb[oneIndex] = 0
                oneIndex+=1
if __name__ == '__main__':    
    main()