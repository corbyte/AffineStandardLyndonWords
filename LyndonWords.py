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
    def __getitem__(self,index):
        return self.order[index]

class rootSystem:
    def commutator(A,B):
        matrix = A.matrix[0]@ B.matrix[0]-B.matrix[0]@ A.matrix[0]
        gcf = np.gcd.reduce(matrix.flatten())
        gcf = max(1,gcf)
        return (np.floor_divide(matrix,gcf,dtype=int), B.matrix[1] + A.matrix[1])
    def __init__(self, ordering,type:str = 'A', n:int = 0, affine:bool =False):
        self.arr = []
        type = type.upper()
        if( len(type) != 1 or type < 'A' or type > 'G' ):
            raise ValueError('Type is invalid')
        if(n == 0):
            if(affine):
                n = len(ordering)-1
            else:
                n = len(ordering)
        self.affine = affine
        if((affine and n!= len(ordering)-1) or (not affine and n!=len(ordering))):
            raise ValueError('Please enter an n that matches the length or ordering')
        self.ordering:letterOrdering = letterOrdering(ordering)
        self.arr.append([word([i],len(self.ordering)) for i in self.ordering.order])
        if(affine):
            if(type == 'A'):
                self.delta = rootSystem.TypeADelta(n)
            elif (type == 'B'):
                self.delta = rootSystem.TypeBDelta(n) 
            elif(type == 'C'):
                self.delta = rootSystem.TypeCDelta(n)
            
    def getWords(self, combination):
        sameLengths = self.arr[sum(combination)-1]
        word = []
        for i in sameLengths:
            if(np.all(i.weights == combination)):
                word.append(i)
        return word
    def getAffineWords(self,weight):
        if not self.affine:
            raise ValueError('Cannot call getAffineWords on a simple Lie algebra')
        matches = []
        for k in range(int((len(self.arr)-int(np.sum(weight)))/sum(self.delta)+1)):
            matches.extend(self.getWords(weight + k*self.delta))
        return matches
    def isImaginary(self,combinations):
        return (sum(combinations) % sum(self.delta) == 0)
    def genWord(self, combinations):
        combinations = np.array(combinations,dtype=int)
        weight = sum(combinations)
        if(self.affine):
            imaginary = self.isImaginary(combinations)
        else:
            imaginary = False
        potentialOptions = []
        if (weight > len(self.arr)+1):
            return None
        minSubRoot  = self.arr[0][0]
        for i in self.arr[0]:
            if i < minSubRoot:
                minSubRoot = i
        for i in range(1,weight//2 + 1):
            minlen = self.arr[i-1]
            for j in minlen:
                diff = combinations - j.weights
                if((min(diff) < 0) or (not imaginary and j < minSubRoot)):
                    continue
                complement = self.getWords(diff)
                if len(complement) == 0:
                    continue
                for comp in complement:
                    if(comp < j):
                        if(not imaginary and comp < minSubRoot):
                            continue
                        newWord = comp + j
                        if(self.affine and (self.isImaginary(comp.weights) or self.isImaginary(j.weights))):
                            bracket = rootSystem.commutator(comp,j)
                            #Checks to see if bracket is non-zero
                            if not bracket[0].any():
                                continue
                            newWord.matrix = bracket
                        minSubRoot = comp
                        potentialOptions.append((newWord,comp,j))
                    else:
                        newWord = j + comp
                        if(self.affine and (self.isImaginary(comp.weights) or self.isImaginary(j.weights))):
                            bracket = rootSystem.commutator(j,comp)
                            if not bracket[0].any():
                                continue
                            newWord.matrix = bracket
                        minSubRoot = j
                        potentialOptions.append((newWord,j,comp))
        if(len(self.arr) < weight):
            self.arr.append([])
        if not imaginary:
            match = potentialOptions[-1][0]
            if(self.affine):
                match.matrix = rootSystem.commutator(potentialOptions[-1][1],potentialOptions[-1][2])
            self.arr[weight-1].append(match)
            return match
        else:
            potentialOptions = list(set(potentialOptions))
            potentialOptions.sort(reverse=True)
            while(not rootSystem.commutator(potentialOptions[0][1],potentialOptions[0][2])[0].any()):
                potentialOptions.pop(0)
            potentialOptions[0][0].matrix = rootSystem.commutator(potentialOptions[0][1],potentialOptions[0][2])
            liPotentialOptions = [potentialOptions[0][0]]
            liset = [potentialOptions[0][0].matrix[0].flatten()]
            index = 1
            while(len(liset) < len(self.ordering)-1):
                #change to only use non-zero matrix entries
                potentialOptions[index][0].matrix = rootSystem.commutator(potentialOptions[index][1],potentialOptions[index][2])
                liprime = liset + [potentialOptions[index][0].matrix[0].flatten()]
                if(np.linalg.matrix_rank(np.vstack(liprime)) == len(liprime)):
                    liset = liprime
                    liPotentialOptions.append(potentialOptions[index][0])
                index += 1
            self.arr[-1].extend(liPotentialOptions)
            return liPotentialOptions
    def getBaseWeights(self):
        returnarr = []
        for i in range(len(self.arr)):
            for j in self.arr[i]:
                returnarr.append(j.weights)
            if( i == sum(self.delta) -1 ):
                returnarr.append(self.delta)
                return returnarr
    def getWordsByBase(self):
        returnarr = []
        for i in self.getBaseWeights():
            returnarr.append(self.getAffineWords(i))
        return returnarr
    def checkMonotonicity(self):
        returnarr = []
        for i in self.getBaseWeights()[:-1]:
            weights = [i.weights for i in self.getAffineWords(i)]
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
            returnarr.append((str(weights[0]),monotonicity))
        return returnarr
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
    if type == "A":
        Atype(order,affineCount)
    elif type == "B":
        Btype(order,affineCount)
    elif type == "C":
        Ctype(order,affineCount)
    elif type == "D":
        Dtype(order,affineCount)

def Btype(ordering,affineCount=0):
    return genTypeBFinite(ordering,True)
def Ctype(ordering,affineCount=0):
    if(affineCount == 0):
        sLyndonWords = genTypeCFinite(ordering,True)
    else:
        sLyndonWords = genTypeCAffine(ordering,affineCount,True)
    return sLyndonWords
def Dtype(ordering,affineCount=0):
    return genTypeDFinite(ordering,printIt=True)
def Atype(ordering,affineCount=0):
    if(affineCount == 0):
        sLyndonWords = genTypeAFinite(ordering,printIt=True)
    else:
        sLyndonWords = genTypeAAffine(ordering,affineCount,True)
    return sLyndonWords
def genTypeBFinite(ordering,printIt=False):
    BRootSystem = rootSystem(ordering,'B')
    size = len(ordering)
    for length in range(2,2*size):
        #i
        if(length <= size):
            comb = np.zeros(size,dtype=int)
            for i in range(size-length,size):
                comb[i] = 1
            if printIt:
                print(BRootSystem.genWord(comb))
            else:
                BRootSystem.genWord(comb)
            if(length != size):
                #ei - ej
                for start in range(0,size - length):
                    comb = np.zeros(size,dtype=int)
                    for k in range(start,start+length):
                        comb[k] = 1
                    if printIt:
                        print(BRootSystem.genWord(comb))
                    else:
                        BRootSystem.genWord(comb)
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
                    print(BRootSystem.genWord(comb))
                else:
                    BRootSystem.genWord(comb)
                twoIndex -= 1
                comb[twoIndex] = 2
                comb[oneIndex] = 0
                oneIndex+=1
    return BRootSystem
        
def genTypeCFinite(ordering,printIt=False):
    size=len(ordering)
    CRootSystem = rootSystem(ordering,'C')
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
                    print(CRootSystem.genWord(comb))
                else:
                    CRootSystem.genWord(comb)
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
                    print(CRootSystem.genWord(comb))
                else:
                    CRootSystem.genWord(comb)
        #2i
        if(length % 2 == 1):
            comb=np.zeros(size,dtype=int)
            comb[-1] = 1
            for k in range(size-2,size-2-length//2,-1):
                comb[k] = 2
            if printIt:
                print(CRootSystem.genWord(comb))
            else:
                CRootSystem.genWord(comb)
    return CRootSystem
def genTypeCAffine(ordering,affineCount,printIt=False)-> rootSystem:
    size=len(ordering)
    CRootSystem = rootSystem(ordering,'C',affine=True)
    for i in range(len(CRootSystem.arr[0])):
        rootIndex = CRootSystem.arr[0][i][0].rootIndex
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
        CRootSystem.arr[0][i].matrix = (matrix,t)
    delta = CRootSystem.delta
    simpleLetterOrdering = genTypeCFinite([i for i in range(1,size)]).arr
    weights = [None]*len(simpleLetterOrdering)
    for row in range(len(simpleLetterOrdering)):
        weights[row] = [None]*len(simpleLetterOrdering[row])
        for root in range(len(simpleLetterOrdering[row])):
            weights[row][root] = simpleLetterOrdering[row][root].weights.tolist()+ [0]
    for deltaCount in range(affineCount+1):
        if(deltaCount > 0):
            if printIt:
                print(*CRootSystem.genWord(deltaCount*delta),sep='\n')
            else:
                CRootSystem.genWord(deltaCount*delta)
        for length in range(1,2*size-2):
            if(deltaCount > 0 or length > 1):
                for comb in weights[length-1]:
                    if printIt:
                        print(CRootSystem.genWord(comb+deltaCount*delta))
                    else:
                        CRootSystem.genWord(comb+deltaCount*delta)
                for comb in weights[-length]:
                    if printIt:
                        print(CRootSystem.genWord((deltaCount+1)*delta - comb))
                    else:
                        CRootSystem.genWord((deltaCount+1)*delta - comb)
    return CRootSystem
            
        
def genTypeAFinite(ordering,printIt=False):
    size = len(ordering)
    ARootSystem = rootSystem(ordering)
    for length in range(2,size+1):
        for start in range(0,size - length + 1):
            comb = np.zeros(size,dtype=int)
            for k in range(start,start+length):
                comb[k] = 1
            if printIt:
                print(ARootSystem.genWord(comb))
            else:
                ARootSystem.genWord(comb)
    return ARootSystem
def genTypeAAffine(ordering,affineCount,printIt=False):
    size = len(ordering)
    ARootSystem = rootSystem(ordering, affine=True)
    for i in range(len(ARootSystem.arr[0])):
        rootIndex = ARootSystem.arr[0][i][0].rootIndex
        if(rootIndex == 0):
            arr = np.zeros((size,size),dtype=int)
            arr[-1,0] = -1
            t=1
        else:
            arr = np.zeros((size,size),dtype=int)
            arr[rootIndex-1,rootIndex] = 1
            t=0
        ARootSystem.arr[0][i].matrix = (arr,t)
    delta = ARootSystem.delta
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
                        for i in ARootSystem.genWord(comb):
                            print(f"{i} {comb}")
                    else:
                        ARootSystem.genWord(comb)
                else:
                    if printIt:
                        print(f"{ARootSystem.genWord(comb)} {comb}")
                    else:
                        ARootSystem.genWord(comb)
    return ARootSystem
def genTypeDFinite(ordering,printIt=False):
    size=len(ordering)
    DRootSystem = rootSystem(ordering,'D')
    for length in range(2,2*size-2):
        #i-j
        for start in range(0,size - length):
            comb = np.zeros(size,dtype=int)
            for k in range(start,start+length):
                comb[k] = 1
            if printIt:
                print(DRootSystem.genWord(comb))
            else:
                DRootSystem.genWord(comb)
        #i+j
        if(length < size):
            comb=np.zeros(size,dtype=int)
            comb[-1] =1
            for i in range(-(length+1),-2,1):
                comb[i] = 1
            if printIt:
                print(DRootSystem.genWord(comb))
            else:
                DRootSystem.genWord(comb)
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
                    print(DRootSystem.genWord(comb))
                else:
                    DRootSystem.genWord(comb)
                twoIndex -= 1
                comb[twoIndex] = 2
                comb[oneIndex] = 0
                oneIndex+=1
if __name__ == '__main__':    
    main()