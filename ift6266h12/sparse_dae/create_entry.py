"""
load the trained model saved every epoch.
load the test set
transform the test set by the trained model
pca on the transformation
create the submission
"""

import scipy as sp
import numpy as np
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
import pylearn2.utils.serial as serial
from  pylearn2.base import Block
#from toy_datasets import ToyDatasetSparse
import theano.sparse
import theano
from terry import Terry

def load_model(model_path):
    model = serial.load(model_path)
    return model

def load_inputs():
    terry =  Terry()
    return terry.testset

def get_filter(model):
    X = theano.sparse.csr_matrix()
    Y = model.encode(X)
    F = theano.function([X], Y)
    return F
    
def main():
    import ipdb; ipdb.set_trace()
    
    # reload the model of the specific epoch
    model = load_model('terry_30_epoch.pkl')
    # get the sparse dataset that will be transformed
    # by the model
    inputs = load_inputs()
    # compile a theano transforming function
    filter = get_filter(model)
    # transform the inputs to outputs
    outputs = filter(inputs.sparse_matrix)
    

    # pca with scikit learn
    q = 4
    pca = PCA(n_components=q)
    pca.fit(outputs)
    results = pca.transform(outputs)

    # finally preparing the submission
    numpy.savetxt('terry_pca4_final.prepro', results)
    
if __name__ == '__main__':
    main()
