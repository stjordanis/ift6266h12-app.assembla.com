import theano.tensor
import theano.sparse
import numpy.random
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError

class SampledMeanSquaredReconstructionError(MeanSquaredReconstructionError):
    def __init__(self):
        self.random_stream = RandomStreams(seed=1)

    def __call__(self, model, X):
        # X is theano sparse
        X_dense=theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1, prob=0.5, ndim=None)
        
        # a random pattern that indicates to reconstruct all the 1s and some of the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P>0, 1, 0)

        # penalty on activations
        L1_units = theano.tensor.abs_(model.encode(X)).sum()
        
        # penalty on weights
        params = model.get_params()
        W = params[2]
        L1_weights = theano.tensor.abs_(W).sum()
        
        cost = (((model.reconstruct(X, P) - X_dense * P) ** 2).sum(axis=1).mean()
                + 0.001 * (L1_weights + L1_units))

        return cost
