import numpy

def uniformization(inparray,zer = True):
    "Exact uniformization of the inparray (matrix) data"
    # Create ordered list of elements
    listelem = list(numpy.sort(list(set(inparray.flatten()))))
    dictP = {}
    totct = 0
    outarray = numpy.ones_like(inparray)
    #initialize element count
    for i in listelem:
        dictP.update({i:0})
    #count
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                dictP[inparray[i,j]]+=1
                totct +=1
        else:
            dictP[inparray[i]]+=1
            totct +=1
    #cumulative
    prev = 0
    for i in listelem:
        dictP[i]+= prev
        prev = dictP[i]
    #conversion
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                outarray[i,j] = dictP[inparray[i,j]]/float(totct)
        else:
            outarray[i] = dictP[inparray[i]]/float(totct)
    if zer:
        outarray = outarray - dictP[listelem[0]]/float(totct)
        outarray /= outarray.max()
    return outarray
