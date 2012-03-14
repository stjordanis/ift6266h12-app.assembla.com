import numpy
import scipy.sparse
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sparse_dataset import SparseDataset
import numpy.random

class ToyDatasetDense(DenseDesignMatrix):
    # dense sets should inherits directly from DenseDesignMatrix
    # while sparse sets inherits from SparseDataset.
    def __init__(self):

        # simulated random dataset
        rng = numpy.random.RandomState(seed=42)
        data = rng.normal(size=(1000, 10))
        self.y = numpy.random.binomial(1, 0.5, [1000, 1])
        super(ToyDatasetDense, self).__init__(data)

class ToyDatasetSparse(SparseDataset):
    def __init__(self):
        data = scipy.sparse.eye(2000, 200, k=0, format='csr')
        super(ToyDatasetSparse, self).__init__(from_scipy_sparse_dataset=data)
