import os
import pyter

def getLabels(labelFile):
    f = open(labelFile, 'r')
    wavFiles = []
    curFileName = ""
    curFile = []
    for line in f:
        line = line.split()
        if line[2] == curFileName:
            curFile += [line]
        else:
            curFileName = line[2]
            wavFiles += [curFile]
            curFile = []

    wavFiles = wavFiles[1:]

    output = []
    for wav in wavFiles:
        labels = [wav[0][0]]
        currLab = wav[0][0]
        prevLab = wav[0][0]
        correctLabels = [wav[0][1]]
        currCorrect = wav[0][1]
        prevCorrect = wav[0][1]
        ctLab = 1
        ctCorr = 1
        for x in range(2, len(wav)-2):
            if (labels[-1] != wav[x][0] and ((wav[x-2][0] == wav[x][0]) or (wav[x-1][0] == wav[x][0])) and ((wav[x+1][0] == wav[x][0]) or (wav[x+2][0] == wav[x][0]))):
                labels += [wav[x][0]]
            if correctLabels[-1] != wav[x][1]:
                correctLabels += [wav[x][1]]
        output += [[labels, correctLabels, wav[0][2]]]

    return output

l = getLabels('MLPTrainout.txt')

def wer(ref, hyp ,debug=False):
    r = ref
    h = hyp
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    SUB_PENALTY = 1
    INS_PENALTY = 1
     
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
         
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
     
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                 
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    #return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}




xs = 0.0
s = 0.0
d = 0.0
i = 0.0
n = 0.0
for x in l:
	xs += 1
	w = wer(x[1], x[0], False)
	s += w['Sub']
	i += w['Ins']
	d += w['Del']
	n += len(x[1])

print ""
print (n - s - d - i)/n