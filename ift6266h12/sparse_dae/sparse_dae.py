import numpy
import theano
import theano.sparse
from theano import tensor
import sampling_dot
#from theano.sparse.sandbox.sp2 import sampling_dot
from pylearn2.autoencoder import DenoisingAutoencoder
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class Rectify(object):
    def __call__(self, X_before_activation):
        # X_before_activation is linear inputs of hidden units, dense
        return X_before_activation * (X_before_activation > 0)

class SparseVectorSpace(VectorSpace):
    
    def make_theano_batch(self, name = None, dtype = None):
        if dtype is None:
            dtype = theano.config.floatX
            
        return theano.sparse.csr_matrix(name = name)

        
class SparseDenoisingAutoencoder(DenoisingAutoencoder):

    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001):

        # sampling dot only supports tied weights
        assert tied_weights == True
                
        super(SparseDenoisingAutoencoder, self).__init__(corruptor,
                                    nvis, nhid, act_enc, act_dec,
                                    tied_weights=False, irange=1e-3, rng=9001)
        
        self.input_space = SparseVectorSpace(nvis)
        
    def get_params(self):
        # this is needed because sgd complains when not w_prime is not used in grad
        # so delete w_prime from the params list
        params = super(SparseDenoisingAutoencoder, self).get_params()
        return params[0:3]
        
    def _initialize_weights(self, nvis, rng=None, irange=None):
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        # TODO: use weight scaling factor if provided, Xavier's default else
        self.weights = sharedX(
            (.5 - rng.rand(nvis, self.nhid)) * irange,
            name='W',
            borrow=True
        )
        # for debugging
        self.weights.tag.test_value = (.5 - rng.rand(nvis, self.nhid)) * irange

    def _initialize_hidbias(self):
        self.hidbias = sharedX(
            numpy.zeros(self.nhid),
            name='hb',
            borrow=True
        )
        self.hidbias.tag.test_value = numpy.zeros(self.nhid)

    def _initialize_visbias(self, nvis):
        self.visbias = sharedX(
            numpy.zeros(nvis),
            name='vb',
            borrow=True
        )
        self.visbias.tag.test_value = numpy.zeros(nvis)

    def _initialize_w_prime(self, nvis, rng=None, irange=None):
        assert not self.tied_weights, (
            "Can't initialize w_prime in tied weights model; "
            "this method shouldn't have been called"
        )
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        self.w_prime = sharedX(
            (.5 - rng.rand(self.nhid, nvis)) * irange,
            name='Wprime',
            borrow=True
        )
        self.w_prime.tag.test_value = (.5 - rng.rand(self.nhid, nvis)) * irange
    
    def encode(self, inputs):
        if (isinstance(inputs, theano.sparse.basic.SparseVariable)
            or (isinstance(inputs, theano.tensor.Variable))):
            return self._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def decode(self, hiddens, pattern):
        """
        Map inputs through the encoder function.

        Parameters
        ----------
        hiddens : tensor_like or list of tensor_likes
        Theano symbolic (or list thereof) representing the input
        minibatch(es) to be encoded. Assumed to be 2-tensors, with the
        first dimension indexing training examples and the second indexing
        data dimensions.

        pattern: theano.matrix, the same shape of the minibatch inputs
        0/1 like matrix specifying how to reconstruct inputs. 
        
        Returns
        -------
        decoded : tensor_like or list of tensor_like
        Theano symbolic (or list thereof) representing the corresponding
        minibatch(es) after decoding.
        """
        if self.act_dec is None:
            act_dec = lambda x: x
        else:
            act_dec = self.act_dec
            if isinstance(hiddens, tensor.Variable):
                
                return act_dec(self.visbias + sampling_dot.sampling_dot(hiddens, self.weights, pattern))
            else:
                return [self.decode(v, pattern) for v in hiddens]

    def reconstruct(self, inputs, pattern):
        
        return self.decode(self.encode(inputs), pattern)
    
    def _hidden_input(self, x):
        """
        Given a single minibatch, computes the input to the
        activation nonlinearity without applying it.
        
        Parameters
        ----------
        x : theano sparse variable 
        Theano symbolic representing the corrupted input minibatch.
        
        Returns
        -------
        y : tensor_like
        (Symbolic) input flowing into the hidden layer nonlinearity.
        """
        # after corruption, x is still sparse
                
        return self.hidbias + theano.sparse.dot(x, self.weights)
    