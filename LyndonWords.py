import numpy as np
class letter:
    index:int
    value:str
    rootIndex:int
    def __init__(self, v:str="",root:int=0):
        self.value = v
        self.rootIndex = root
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
    def toWeights(self,l):
        weights = np.zeros(l,dtype=int)
        for i in self.string:
            weights[i.index] += 1
        return weights
    def __init__(self, wordArray):
        self.string = np.array(wordArray,dtype=letter)
    def __str__(self):
        k = ""
        for i in self.string:
            k += i.value
        return k
    def __eq__(self,other):
        if(len(self.string) != len(other.string)):
            return False
        for i in range(len(self.string)):
            if(self.string[i] != other.string[i]):
                return False
        return True
    def __lt__(self,other):
        for i in range(min(len(self.string),len(other.string))):
            if(self.string[i] > other.string[i]):
                return False
        if(len(self.string) <= len(other.string)):
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
        return word(np.concatenate((self.string,other.string),dtype=word)) 
class letterOrdering:
    letterOrdering:list[letter]
    def __init__(self, letterOrdering):
        self.letterOrdering = letterOrdering
        for i in range(len(letterOrdering)):
            self.letterOrdering[i].index = i
class standardLyndonWords:
    arr = []
    ordering:letterOrdering
    def __init__(self, ordering):
        self.arr.append([word([i]) for i in ordering.letterOrdering])
        self.ordering = ordering
    def getWord(self, combination):
        sameLengths = self.arr[sum(combination)-1]
        for i in sameLengths:
            if(np.all(i.toWeights(len(self.ordering.letterOrdering)) == combination)):
                return i
        return None
    def genWord(self, combinations):
        weight = sum(combinations)
        potentialOptions = []
        if (weight > len(self.arr)+1):
            return None
        for i in range(1,weight//2 + 1):
            minlen = self.arr[i-1]
            for j in minlen:
                diff = list(map(lambda a, b: a-b, combinations,j.toWeights(len(self.ordering.letterOrdering))))
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
        m = max(potentialOptions)
        self.arr[weight-1].append(m)
        return m
def parseWord(s:str):
    if(len(s) == 1):
        return s
    rightFactor = s[-1]
    for i in range(len(s)-1,0,-1):
        if(s[i]<rightFactor[0]):
            return
def main():
    print("Select group:")
    print("A")
    group = input()
    match group:
        case "A":
            Agroup()
        case "a":
            Agroup()

def Agroup():
    print("Enter size:")
    size = int(input())
    ordered = np.empty(size,dtype=letter)
    for i in range(size):
        print(f"Enter root index for {i} ordered element")
        index = input()
        ordered[i]= letter(index,int(index))
    ordering = letterOrdering(ordered)
    sLyndonWords = standardLyndonWords(ordering)
    for i in range(2,size+1):
        for j in range(0,size - i + 1):
            comb = np.zeros(size,dtype=int)
            for k in range(j,j+i):
                comb[k] = 1
            print(sLyndonWords.genWord(comb))
if __name__ == '__main__':
    main()