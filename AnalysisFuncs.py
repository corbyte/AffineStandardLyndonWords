from LyndonWords import *

def standard_delta_pattern(rootsys:rootSystem,index:int) -> bool:
    deltaWords = rootsys.get_chain(rootsys.delta)[index::rootsys.n]
    leftFac,rightFac = rootsys.standfac(deltaWords[0])
    for i in deltaWords[1:]:
        if(word.letter_list_cmp(i[:len(leftFac)],leftFac.string) != 0):
            return False
        parsed_form = rootsys.parse_to_block_format(i[len(leftFac):],include_flipped=False)
        if(len(parsed_form) != 2):
            return False
        if(word.letter_list_cmp(parsed_form[1].get_letter_arr(), rightFac.string) != 0):
            return False
        if(parsed_form[0].get_type() != 'im'):
            return False
    return True 

def flipped_delta_pattern(rootsys: rootSystem, index: int) ->bool:
    deltaWords = rootsys.get_chain(rootsys.delta)[index::rootsys.n]
    leftFac,rightFac = rootsys.standfac(deltaWords[0])
    for i in deltaWords[1:]:
        if(word.letter_list_cmp(i[:len(leftFac)],leftFac.string) != 0):
            return False
        parsed_form = rootsys.parse_to_block_format(i[len(leftFac):],True)
        if(len(parsed_form) != 2):
            return False
        if(word.letter_list_cmp(parsed_form[1].get_letter_arr(), rightFac.string) != 0):
            return False
        if(parsed_form[0].get_type() != 'pim'):
            return False
    return True 

def definitive_delta_pattern(rootsys: rootSystem, index: int)-> bool:
    deltaWords = rootsys.get_chain(rootsys.delta)[index::rootsys.n]
    leftFac,rightFac = rootsys.standfac(deltaWords[0])
    for i in deltaWords[1:]:
        if(word.letter_list_cmp(i[:len(leftFac)],leftFac.string) != 0):
            return False
        parsed_form = rootsys.parse_to_block_format(i[len(leftFac):],True)
        if(len(parsed_form) != 2):
            return False
        if(word.letter_list_cmp(parsed_form[1].get_letter_arr(), rightFac.string) != 0):
            return False
        if(parsed_form[0].get_type() != 'pim'):
            return False
    return True 
class deltaTypes:
    def __init__(self,index:int,hs:list,type:str,insertedIndex:int,leftfac:str,rightfac:str, flipIndex = 0):
        self.index = index
        self.hs = hs
        #is either "last" or "cofac" or "neither" if something goes horribly wrong
        self.type = type
        self.insertedIndex = insertedIndex
        self.leftfac = leftfac
        self.rightfac = rightfac
        self.flipIndex = flipIndex
    def to_list(self):
        return [self.index,self.type,self.insertedIndex,self.leftfac,self.rightfac,self.flipindex]
class deltaTypesCollection:
    def __init__(self,rootsys:rootSystem,deltaTypesList):
        self.type = rootsys.type
        self.ordering = str(rootsys.ordering)
        self.deltaTypes = deltaTypesList
    def to_csv(self)->str:
        retstr = f"{self.type},{self.ordering},,,,,\n"
        for i in self.deltaTypes:
            retstr+= f"{i.index},{i.type},{i.insertedIndex},{i.leftfac},{i.rightfac},{i.flipIndex},[{' '.join([str(j) for j in i.hs])}]\n"
        return retstr
    def not_all_standard(self) -> bool:
        for i in self.deltaTypes:
            if i.type != "last":
                return True
        return False
def generate_delta_types(rootsys:rootSystem,k=2) ->deltaTypesCollection:
    deltaWords = rootsys.SL(rootsys.delta)
    listOfDeltaTypes = []
    kdeltaWords = rootsys.SL(rootsys.delta*k)
    for i in range(rootsys.n):
        kdeltaWord = kdeltaWords[i]
        if(standard_delta_pattern(rootsys,i)):
            breakType = "standard"
        elif(flipped_delta_pattern(rootsys,i)):
            breakType = "flipped"
        else:
            breakType = "other"
        factors = rootsys.standfac(deltaWords[i])
        block = rootsys.parse_to_block_format(kdeltaWord[-rootsys.deltaHeight - len(factors[1]):-len(factors[1])],True)[0]
        flip_index = block.get_perm_index()
        inserted_index = block.get_index()
        listOfDeltaTypes.append(deltaTypes(i+1,deltaWords[i].hs,breakType,inserted_index,factors[0].no_commas(),factors[1].no_commas(),flip_index))
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
    if(word.letter_list_cmp(rootsys.SL(rootsys.delta - minusarr)[0].string,deltaWords[index][:-1]) != 0):
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
def flipped_delta_type_conditions_met(rootsys:rootSystem,index:int) -> bool:
    SLi = rootsys.SL(rootsys.delta)[index]
    lastLetterIndex = SLi[-1].rootIndex
    SLprime = np.copy(rootsys.delta)
    SLprime[lastLetterIndex] -= 1
    if(word.letter_list_cmp(SLi[:-1],rootsys.SL(SLprime)[0].string)==0):
        return False
    leftfac,rightfac = rootsys.costfac(SLi)
    for i in range(len(rightfac)-1):
        if(leftfac[i] != rightfac[i]):
            return False
    return True
def check_delta_type_prediction(rootsys:rootSystem,k=2):
    #Depreciated
    result = generate_delta_types(rootsys,k)
    deltaWords = rootsys.SL(rootsys.delta)
    for i in range(len(result.deltaTypes)):
        if((result.deltaTypes[i].type == "cofac") != costfac_delta_type_conditions_met(rootsys,deltaWords,i)):
            yield (rootsys.type, str(rootsys.ordering),i+1,result.deltaTypes[i].type,"cofac")
        #if((result.deltaTypes[i].type == "last") != lastSplitDeltaTypeConditionsMet(rootsys,deltaWords,i)):
        #    exceptions.append((rootsys.type, str(rootsys.ordering),i+1,result.deltaTypes[i].type,"last"))
        if((result.deltaTypes[i].type == "flipped") != flipped_delta_type_conditions_met(rootsys,i)):
           yield (rootsys.type, str(rootsys.ordering),i+1,result.deltaTypes[i].type,"flipped") 
def check_delta_type_prediction_perms(rootsystems,k=2):
    rootsys:rootSystem
    for rootsys in rootsystems:
        for i in check_delta_type_prediction(rootsys,k):
            yield i
def check_standard_fac_same(rootsys:rootSystem,k=2):
    rootsys.generate_up_to_delta(k)
    baseFacs = [rootsys.standfac(i)[0] for i in rootsys.SL(rootsys.delta)]
    for i in range(len(baseFacs)):
        others = [rootsys.standfac(i)[0] for i in 
                  rootsys.get_chain(rootsys.delta)[i::len(baseFacs)]]
        for o in others:
            if(not np.all(o.degree-(o.height//(rootsys.deltaHeight))*rootsys.delta== baseFacs[i].degree)):
                return (str(rootsys.ordering),i,o.degree-(o.height//(rootsys.deltaHeight)),baseFacs[i].degree)
    return True
class MaxPeriodicityReturn:
    def __init__(self,periodicity,maxRoot,ordering):
        self.periodicity = periodicity
        self.maxRoot = maxRoot
        self.ordering = ordering
def max_periodicity_rootSystem(rootsys:rootSystem):
    max = 0
    for i in rootsys.baseRoots[:-1]:
        print(i)
        res = rootsys.periodicity(i)
        if(res > max):
            max = res
            maxRoot = i
    return MaxPeriodicityReturn(max,maxRoot,str(rootsys.ordering))
def check_basic_periodicity(rootSys:rootSystem,k=2):
    imwords = rootSys.SL(rootSys.delta)
    imfacs = [rootSys.costfac(i)[0] for i in imwords] + [rootSys.costfac(i)[1] for i in imwords]
    for i in rootSys.baseRoots:
        if(len(rootSys.SL(i+k*rootSys.delta)[0]) == 2):
            flag = False
            for imfac in imfacs:
                if(imfac == rootSys.SL(i)[0]):
                    flag = True
                    break
            if(not flag):
                return i
def verify_periodicity(rootSys:rootSystem):
    for _ in rootSys.verify_periodicity():
        return False
    return True
class lca_critical_roots_return_class:
    def __init__(self,rootsysorder,w,leftcrit,rightcrit):
        self.rootsysorder = rootsysorder
        self.word = w
        self.leftcrit = leftcrit
        self.rightcrit = rightcrit
    def __str__(self):
        return str((str(self.rootsysorder),self.word.no_commas(),self.leftcrit.no_commas(),self.rightcrit.no_commas()))
def lca_on_critical_roots(rootSys:rootSystem,k=5):
    for kdelta in range(k+1):
        for baseRoot in rootSys.baseRoots[:-1]:
            criticalPairs = list(rootSys.realized_critical_roots(kdelta *rootSys.delta + baseRoot))
            actual_word = rootSys.SL(kdelta * rootSys.delta + baseRoot)[0]
            for j in range(len(criticalPairs)):
                if(len(word.lca(actual_word,criticalPairs[j][0])) < len(criticalPairs[j][1])):
                    if(word.letter_list_cmp(actual_word.string,list(criticalPairs[j][0].string) + list(criticalPairs[j][1].string)) == 0):
                        continue
                    return lca_critical_roots_return_class(rootSys.ordering,actual_word,criticalPairs[j][0],criticalPairs[j][1])
    return None
def convexity_from_perm(rootsys:rootSystem,k=5,word_convexity=False):
    for i in rootsys.check_convexity(k,word_convexity):
        return (str(rootsys.ordering),i[0].no_commas(),i[1].no_commas(),i[2].no_commas())
    
def periodicity_ell_1(rootSys:rootSystem):
    left_parts = [rootSys.parse_to_block_format(i)[0] for i in rootSys.SL(3*rootSys.delta)]
    for i in rootSys.roots_with_periodicity_n(1):
        parse_block = rootSys.parse_to_block_format(rootSys.get_chain(i)[-1])
        if(len(parse_block) != 3):
            continue
        if parse_block[0].get_type() != 'word' or parse_block[0].get_type() != 'word':
            continue
        if parse_block[0] not in left_parts:
            return str(rootSys.ordering,i)
        return None
    
def check_irr_chains_conj(rootSys:rootSystem,k=3):
    irrchains = rootSys.irr_chains()
    for chain in irrchains:
        w = rootSys.SL(k*rootSys.delta + chain)[0]
        if(len(rootSys.parse_to_delta_format(w)) != 2):
            return (chain,str(rootSys.ordering))
    mod = 0
    for w in rootSys.get_chain(rootSys.delta,k)[rootSys.n:]:
        p = len(w) // rootSys.deltaHeight
        chain_word = rootSys.SL((p-1)*rootSys.delta + irrchains[mod])[0]
        if word.letter_list_cmp(chain_word.string,w.string) != -1:
            return (p,mod+1,w,str(rootSys.ordering))
        mod += 1
        mod %= rootSys.n
    return 