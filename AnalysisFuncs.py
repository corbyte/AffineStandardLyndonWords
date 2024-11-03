from LyndonWords import *
import LyndonWords

def delta_split_at_last(rootsys:rootSystem,index: int) -> bool:
    baseWord = rootsys.get_words(rootsys.delta)[index]
    deltaWords = rootsys.get_affine_words(rootsys.delta)[index + rootsys.n::rootsys.n]
    for deltaWord in deltaWords:
        parsed = rootsys.parse_to_delta_format(deltaWord)
        if(len(parsed) != 3):
            return False
        if(parsed[2] != str(baseWord[-1])):
            return False
    return True

def delta_split_at_cofac(rootsys: rootSystem,index: int) -> bool:
    baseWord = rootsys.get_words(rootsys.delta)[index]
    cofacSplit = rootsys.costfac(baseWord)[0].height
    for deltaWord in rootsys.get_affine_words(rootsys.delta)[index+rootsys.n::rootsys.n]:
        parsed = rootsys.parse_to_delta_format(deltaWord)
        if(len(parsed) != 3):
            return False
        if(parsed[0] != baseWord.no_commas()[:cofacSplit] or parsed[2] != baseWord.no_commas()[cofacSplit:]):
            return False
    return True

def k3_start_delta_pattern(rootsys: rootSystem, index: int) ->bool:
    deltaWords = rootsys.get_affine_words(rootsys.delta)[index::rootsys.n]
    splitting = [i.no_commas() for i in rootsys.costfac(deltaWords[1])]
    if(len(rootsys.parse_to_delta_format(deltaWords[1])) != 1):
        return False
    for i in deltaWords[2:]:
        parsed = rootsys.parse_to_delta_format(i)
        if(len(parsed) != 3):
            return False
        if(parsed[0] != splitting[0] or parsed[2] != splitting[1] or type(parsed[1]) is not list):
            return False
    return True

def last_smallest_delta_pattern(rootsys:rootSystem, index: int) -> bool:
    deltaWords = rootsys.get_affine_words(rootsys.delta)[index::rootsys.n]
    for i in range(-1,-rootsys.deltaHeight-1,-1):
        if(deltaWords[0][i].rootIndex == rootsys.ordering[0].rootIndex):
            smallestIndex = rootsys.deltaHeight + i
            break
    for deltaWord in deltaWords[2:]:
        parsed = rootsys.parse_to_delta_format(deltaWord)
        if(len(parsed) != 3):
            return False
        if(smallestIndex != len(parsed[0])):
            return False
    return True

def two_delta_words_delta_pattern(rootsys:rootSystem,index:int) -> bool:
    deltaWords = rootsys.get_affine_words(rootsys.delta)[index+rootsys.n*2::rootsys.n]
    for deltaWord in deltaWords:
        parsed = rootsys.parse_to_delta_format(deltaWord)
        if(len(parsed) != 4):
            return False
        if(type(parsed[0]) != str or type(parsed[1]) != list or type(parsed[2]) != list or type(parsed[3])!= str):
            return False
    return True
class deltaTypes:
    def __init__(self,index:int,hs:list,type:str,insertedIndex:int,leftfac:str,rightfac:str):
        self.index = index
        self.hs = hs
        #is either "last" or "cofac" or "neither" if something goes horribly wrong
        self.type = type
        self.insertedIndex = insertedIndex
        self.leftfac = leftfac
        self.rightfac = rightfac
    def to_list(self):
        return [self.index,self.type,self.insertedIndex,self.leftfac,self.rightfac]
class deltaTypesCollection:
    def __init__(self,rootsys:rootSystem,deltaTypesList):
        self.type = rootsys.type
        self.ordering = str(rootsys.ordering)
        self.deltaTypes = deltaTypesList
    def to_csv(self)->str:
        retstr = f"{self.type},{self.ordering},,,,\n"
        for i in self.deltaTypes:
            retstr+= f"{i.index},[{' '.join([str(j) for j in i.hs])}],{i.type},{i.insertedIndex},{i.leftfac},{i.rightfac}\n"
        return retstr
    def not_all_last(self) -> bool:
        for i in self.deltaTypes:
            if i.type != "last":
                return True
        return False
def generate_delta_types(rootsys,k=3) ->deltaTypesCollection:
    deltaWords = rootsys.get_words(rootsys.delta)
    listOfDeltaTypes = []
    kdeltaWords = rootsys.get_words(rootsys.delta*k)
    for i in range(rootsys.n):
        kdeltaWord = kdeltaWords[i]
        splitting = 0
        parsedForm = rootsys.parse_to_delta_format(kdeltaWord)
        for j in parsedForm:
            if(type(j) is not str):
                splitting = j[0]
                break
        if(delta_split_at_last(rootsys,i)):
            breakType = "last"
        elif(delta_split_at_cofac(rootsys,i)):
            breakType = "cofac"
        elif(k3_start_delta_pattern(rootsys,i)):
            breakType = "k3start"
        elif(last_smallest_delta_pattern(rootsys,i)):
            breakType = "lastSmallest"
        elif(two_delta_words_delta_pattern(rootsys,i)):
            breakType = "twoDeltaWords"
        else:
            breakType = "other"
        factors = rootsys.costfac(deltaWords[i])
        listOfDeltaTypes.append(deltaTypes(i+1,deltaWords[i].hs,breakType,splitting,factors[0].no_commas(),factors[1].no_commas()))
    return deltaTypesCollection(rootsys,listOfDeltaTypes)
def costfac_delta_type_conditions_met(rootsys:rootSystem,deltaWords,index):
    factorization = rootsys.costfac(deltaWords[index])
    for i in range(index):
        if(factorization[1] < deltaWords[i]):
            if(rootsys.eBracket(rootsys.costfac(deltaWords[index])[1] + deltaWords[i])):
                return True
        else:
            break
    minusarr = np.zeros(rootsys.n+1,dtype=int)
    minusarr[deltaWords[index][-1].rootIndex] += 1
    if(word.letter_list_cmp(rootsys.get_words(rootsys.delta - minusarr)[0].string,deltaWords[index][:-1]) != 0):
        return True
    for j in range(index):
        if(word.letter_list_cmp(rootsys.costfac(deltaWords[j])[0].string,
                        rootsys.costfac(deltaWords[index])[1].string[:-1]) == 0):
            return True
    return False
def last_split_delta_type_conditions_met(rootsys:rootSystem,deltaWords:list,index:int) -> bool:
    if(rootsys.delta[rootsys.ordering[0].rootIndex] == 1):
        return True
    rightfac = rootsys.costfac(deltaWords[index])[1]
    for deltaWord in deltaWords[:index]:
        if(word.letter_list_cmp(rootsys.costfac(deltaWord)[1].string,rightfac.string[:-1]) == 0):
            return True
    return False
def k3_start_delta_type_conditions_met(rootsys:rootSystem,index:int) -> bool:
    SLi = rootsys.get_words(rootsys.delta)[index]
    lastLetterIndex = SLi[-1].rootIndex
    SLprime = np.copy(rootsys.delta)
    SLprime[lastLetterIndex] -= 1
    if(word.letter_list_cmp(SLi[:-1],rootsys.get_words(SLprime)[0].string)==0):
        return False
    leftfac,rightfac = rootsys.costfac(SLi)
    for i in range(len(rightfac)-1):
        if(leftfac[i] != rightfac[i]):
            return False
    return True
def check_delta_type_prediction(rootsys:rootSystem,k=2):
    result = generate_delta_types(rootsys,k)
    deltaWords = rootsys.get_words(rootsys.delta)
    for i in range(len(result.deltaTypes)):
        if((result.deltaTypes[i].type == "cofac") != costfac_delta_type_conditions_met(rootsys,deltaWords,i)):
            yield (rootsys.type, str(rootsys.ordering),i+1,result.deltaTypes[i].type,"cofac")
        #if((result.deltaTypes[i].type == "last") != lastSplitDeltaTypeConditionsMet(rootsys,deltaWords,i)):
        #    exceptions.append((rootsys.type, str(rootsys.ordering),i+1,result.deltaTypes[i].type,"last"))
        if((result.deltaTypes[i].type == "k3start") != k3_start_delta_type_conditions_met(rootsys,i)):
           yield (rootsys.type, str(rootsys.ordering),i+1,result.deltaTypes[i].type,"k3start") 
def check_delta_type_prediction_perms(rootsystems,k=2):
    rootsys:rootSystem
    for rootsys in rootsystems:
        for i in check_delta_type_prediction(rootsys,k):
            yield i
def check_standard_fac_same(rootsys:rootSystem,k=2):
    rootsys.generate_up_to_delta(k)
    baseFacs = [rootsys.standfac(i)[0] for i in rootsys.get_words(rootsys.delta)]
    for i in range(len(baseFacs)):
        others = [rootsys.standfac(i)[0] for i in 
                  rootsys.get_affine_words(rootsys.delta)[i::len(baseFacs)]]
        for o in others:
            if(not np.all(o.weights-(o.height//(rootsys.deltaHeight))*rootsys.delta== baseFacs[i].weights)):
                return (str(rootsys.ordering),i,o.weights-(o.height//(rootsys.deltaHeight)),baseFacs[i].weights)
    return True
class MaxPeriodicityReturn:
    def __init__(self,periodicity,maxRoot,ordering):
        self.periodicity = periodicity
        self.maxRoot = maxRoot
        self.ordering = ordering
def max_periodicity_rootSystem(rootsys:rootSystem):
    max = 0
    for i in rootsys.baseWeights[:-1]:
        res = rootsys.get_periodicity(i)
        if(res > max):
            max = res
            maxRoot = i
    return MaxPeriodicityReturn(max,maxRoot,str(rootsys.ordering))
def monotonicity_conj(rootsys:rootSystem,k=3):
    rootsys.generate_up_to_delta(k)
    for base in rootsys.baseWeights[:-1]:
        baseWord = rootsys.get_words(base)[0]
        monotone = rootsys.get_monotonicity(base)
        for i in rootsys.get_words(rootsys.delta):
            if(i <baseWord and rootsys.e_bracket(i + baseWord)):
                conj_monotone = 1
                break
            if(i > baseWord and rootsys.e_bracket(baseWord + i)):
                conj_monotone = -1
                break
        if(conj_monotone != monotone):
            return (str(rootsys.ordering),base,monotone,conj_monotone)
    return True