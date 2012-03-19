__maintainer__ = "Li Yao"

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy

class ToyDataset(DenseDesignMatrix):
    def __init__(self):
        
       # simulated random dataset
       rng = numpy.random.RandomState(seed=42)
       data = rng.normal(size=(10000, 10))

       super(ToyDataset, self).__init__(data)
        
        
