from pylearn2.datasets.dataset import Dataset
from iteration import SequentialSubsetIterator
import numpy, scipy.sparse
import theano
import gzip
floatX = theano.config.floatX

class SparseDataset(Dataset):
    """
    SparseDataset is by itself an iterator. 
    """
    def __init__(self, load_path=None, from_scipy_sparse_dataset=None):
        
        self.load_path = load_path
        
        if self.load_path != None:
            print 'loading data set...'
            self.sparse_matrix = scipy.sparse.csr_matrix(
                        numpy.load(gzip.open(load_path)), dtype=floatX)      
        else:
            print 'building from given sparse dataset...'
            self.sparse_matrix = from_scipy_sparse_dataset

        self.data_n_rows = self.sparse_matrix.shape[0]
        
    def get_design_matrix(self):
        return self.sparse_matrix
    
    def get_batch_design(self, batch_size, include_labels=False):
        """
        method inherited from Dataset
        """
        self.iterator(mode='sequential', batch_size=batch_size, num_batches=None, topo=None)
        return self.next()
    
    def get_batch_topo(self, batch_size):
        """
        method inherited from Dataset
        """
        raise NotImplementedError('Not implemented for sparse dataset')

    def set_iteration_scheme(self, mode=None, batch_size=None,
                             num_batches=None, topo=False):
        """
        method inherited from Dataset
        """
        self.iterator(mode, batch_size, num_batches, topo)
        
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, rng=None):
        """
        method inherited from Dataset
        """
        self.mode = mode
        self.batch_size = batch_size
        
        if mode == 'sequential':
            self.subset_iterator = SequentialSubsetIterator(self.data_n_rows,
                                            batch_size, num_batches, rng=None)
            return self
        else:
            raise NotImplementedError('Oops!')
        
    
    def __iter__(self):
        return self
    
    def next(self):
        indx = self.subset_iterator.next()
        try:
            mini_batch = self.sparse_matrix[indx]
        except IndexError:
            # the ind of minibatch goes beyond the boundary
            import ipdb; ipdb.set_trace()
        return mini_batch
        
    
