"""
author: Li Yao

Note: pca in scikit is just much faster than the one I have written in pca()
For terry set, PCA is much better than SparsePCA
"""
import scipy as sp
import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
"""
def normalize(data):
    # to make mean = 0 and var = 1
    m = sp.mean(data,axis=0)
    stdev = sp.std(data,axis=0)
    data_normalized = (data - m)/stdev
    return data_normalized

def pca(data,centered):
    # PCA
    '''
    to do: maybe the paper K-means Clustering via Principal Component Analysis, ICML 04
    '''
    # data must be a normalized one
    if not centered:
        m = sp.mean(data,axis=0)
        data = data - m
    # Covariance matrix
    S = np.cov(np.transpose(data))

    # eigen value decomposition
    evals,evecs = np.linalg.eig(S)
    # sort eigen values from max to min
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    return evals, evecs
"""
def main():
    
    f = open('terry_dl30_final.prepro','r')
    data = f.read()
    f.close()

    # from string to numpy ndarray
    data = data.split('\n')
    data_stringList = data[:-1]
    data_list = []
    for i in range(len(data_stringList)):
        sample_str = data_stringList[i]
        sample_int = [int(j) for j in sample_str.split(' ')[:-1]]
        data_list.append(sample_int)

    data = np.array(data_list)
    

    # pca with scikit learn
    q = 4
    pca = PCA(n_components=q)
    pca.fit(data)
    results = pca.transform(data)
    
    np.savetxt('terry_dl30pca4_final.prepro', results)
    
if __name__ == '__main__':
    main()
