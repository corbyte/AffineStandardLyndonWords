class letter:
    index:int
    value:str
    def __int__(v:str):
        value = v
    def set_index(i:int):
        index = i
    def __str__(self):
        return f"({value})"
class lyndonOrdering:
    letterOrdering:list[letter]
    def __int__(letterordering:list[letter]):
        letterOrdering = letterOrdering
        for i in range(len(letterOrdering)):
            letterOrdering[i].index = i
    def cmp(a:letter, b:letter):
        return a.index - b.index
class standardLyndonWords:
    arr = []
    
def parseWord(s:str):
    if(len(s) == 1):
        return s
    rightFactor = s[-1]
    for i in range(len(s)-1,0,-1):
        if(s[i]<rightFactor[0]):
            return
def main():
    print("Enter size:")
    size = input()
    ordering = []
    for i in range(size):
        print(f"Enter root {i}:")
        ordering += input()
    while(True):
        print("Enter string:")
        s = input()
        print(parseWord(s))
    return


if __name__ == '__main__':
    main()