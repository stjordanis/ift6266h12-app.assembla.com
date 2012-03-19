from sparse_dataset import SparseDataset
import scipy.sparse
import numpy
import data_processing

class Terry:
    def __init__(self):

        self.trainset_path = '/data/lisa/data/UTLC/sparse/terry_train.npy.gz'
        self.validset_path = '/data/lisa/data/UTLC/sparse/terry_valid.npy.gz'
        self.testset_path = '/data/lisa/data/UTLC/sparse/terry_test.npy.gz'
        self.use_features_path = '/data/lisa/data/UTLC/sparse/terry_testvalid_activefeat.npy'
        
        self.features_selected = numpy.load(open(self.use_features_path))
        # these are sets before preprocessing
        self.train_set = SparseDataset(load_path=self.trainset_path)
        self.valid_set = SparseDataset(load_path=self.validset_path)
        self.test_set = SparseDataset(load_path=self.validset_path)
        # these are sets after preprocessing
        self.trainset = None
        self.validset = None
        self.testset = None
        #fullset = scipy.sparse.vstack((scipy.sparse.vstack((self.train_set.data, self.valid_set.data)),
        #                              self.test_set.data))
        #self.full_set = SparseDataset(from_sparse_dataset=fullset)
        
        self.pre_processing()
        
    def pre_processing(self):
        print 'preprocessing...'
        
        # uniformization
        raw = numpy.concatenate([self.train_set.sparse_matrix.data,
                                 self.valid_set.sparse_matrix.data,
                                 self.test_set.sparse_matrix.data])

        len_train = len(self.train_set.sparse_matrix.data)
        len_valid = len(self.valid_set.sparse_matrix.data)
        len_test = len(self.test_set.sparse_matrix.data)
        
        out = data_processing.uniformization(raw, False)
        
        self.train_set.sparse_matrix.data = raw[0 : len_train]
        self.valid_set.sparse_matrix.data = raw[len_train : (len_train + len_valid)]
        self.test_set.sparse_matrix.data = raw[-len_test :]

        self.full_train = scipy.sparse.vstack([self.train_set.sparse_matrix,
                                               self.valid_set.sparse_matrix], 'csr')

        # shuffling train set
        self.full_train = self.full_train[numpy.random.permutation(self.full_train.shape[0]), :]

        # feature subset selection
        self.full_train = self.full_train[:, self.features_selected]
        self.valid_set = self.valid_set.sparse_matrix[:, self.features_selected]
        self.test_set = self.test_set.sparse_matrix[:, self.features_selected]
        
        # whitening
        std = numpy.std(self.full_train.data)
        self.full_train /= std
        self.valid_set /= std
        self.test_set /= std

        # finally
        self.trainset = SparseDataset(from_scipy_sparse_dataset=self.full_train)
        self.validset = SparseDataset(from_scipy_sparse_dataset=self.valid_set)
        self.testset = SparseDataset(from_scipy_sparse_dataset=self.test_set)
        
if __name__ == '__main__':
    terry = Terry()
    import pdb;pdb.set_trace()
    
    
