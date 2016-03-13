from collections import namedtuple
from arff import Dataset
DecisionNode = namedtuple('DecisionNode', 'test true false text')

def buildDecisionArray(src, converttoLambda, interpretor):
    # define data for decision nodes
    
    def ismatchedFieldname(text1, text2):
        split1 = text1.split(' ')
        split2 = text2.split(' ')
        return split1[0] == split2[0] and split1[2] == split2[2]
    
    def makeTestFunc(testFuncTxt):
        #if (not convertLambda): return testFuncTxt
        fieldName, operator, value = (testFuncTxt.split(' '))
        exp = 'lambda x : x["{0}"] {1} {2}'.format(fieldName, operator, value)
        if converttoLambda: exp = eval(exp)
        return exp
    
    stack = []
    firstNode = None
    fieldparser = interpretor.fieldparser
    for li, line in enumerate(src):
        nodeInfo = (x.strip() for x in line.split('|') if len(x.strip()) > 0).next()
        if (':' not in line): # not leaf
            # node is an array with three values: testFunc, true statement, false statement, testFuncText
            newnode = [makeTestFunc(nodeInfo), '', '', nodeInfo]
            if (len(stack) == 0):
                stack.append(newnode)
            else:  
                node = stack[-1]
                if (not ismatchedFieldname(node[3], nodeInfo)):
                    if (node[1] == ''):
                        node[1] = newnode
                    else:
                        node[2] = newnode
                        stack.pop()
                    stack.append(newnode)
        else:
            testFunc, label = ([x.strip() for x in nodeInfo.split(':')])
            newnode = [makeTestFunc(testFunc), label, '', testFunc]
            if (len(stack) == 0):
                # node is an array with three values: testFunc, true statement, false statement, testFuncText
                stack.append(newnode)
            else:
                node = stack[-1]
                if (node[1] == ''):
                    node[1] = newnode
                    stack.append(newnode)
                else:
                    node = stack.pop()
                    if (ismatchedFieldname(node[3], testFunc)):
                        node[2] = label
                    else: 
                        node[2] = newnode
                        stack.append(newnode)
        if (firstNode is None):
            firstNode = stack[0]
    return firstNode

def parse(iterobj):
    a_cfm = []
    tree = []
    while (True):
        try:
            line = iterobj.next()
            if (line.startswith('Options:')): 
                strips = line.split(':')
                j45properties = strips[1]
                print 'Options:', j45properties
                continue
            if (line.startswith('------------------')):
                iterobj.next()
                line = iterobj.next()
                while (line != ''):
                    tree.append(line)
                    line = iterobj.next()
                continue
            if (line.startswith('Number of Leaves  :')): 
                strips = line.split(':')
                nLeaves = int(strips[1].strip())
                continue
            if (line == '=== Confusion Matrix ==='):
                iterobj.next()
                iterobj.next()

                arr = [[0 for x in xrange(4)] for x in xrange(4)]
                for i in xrange(4):
                    line = iterobj.next()
                    for j, v in enumerate([int(x) for x in line.split(' ') if x.isdigit()]):
                        arr[i][j] = v
                a_cfm.append(arr)
                continue
        except StopIteration:
            break
    return tree

def convertDecisionNode(data):
    if (type(data) is str): return data
    test, true, false, text =  data
    return DecisionNode(test, convertDecisionNode(true), convertDecisionNode(false), text)
    
def buildDecisionNode(src, interpretor):
    return convertDecisionNode(buildDecisionArray(src, True, interpretor))

def test(decisionNode, treeEvent):
    if (type(decisionNode) is str): return decisionNode
    
    result = decisionNode.test(treeEvent)
    return result and test(decisionNode.true, treeEvent) or test(decisionNode.false, treeEvent)

def translateClass(decisionNode, treeEvent):
    #TODO need an implementation
    """
    decision =  test(decisionNode, treeEvent)
    
    if (decision.startswith('ground')): 
        if (treeEvent.v_sag_a == treeEvent.v_sag_1): return 'AG'
        if (treeEvent.v_sag_b == treeEvent.v_sag_1): return 'BG'
        return 'CG'
    elif (decision.startswith('line')): 
        if (treeEvent.v_sag_a == treeEvent.v_sag_3): return 'BC'
        if (treeEvent.v_sag_b == treeEvent.v_sag_3): return 'AC'
        return 'AB'
    elif (decision.startswith('phase')): return 'ABC'
    
    return 'F'
    """
    pass