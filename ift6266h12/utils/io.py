"""
Input/Output functionality
"""
import theano
import numpy
import os

def load_npy(path):
    """
    Function to load npy files. The single difference from `numpy.load` is
    that, if the loaded ndarray has dtype `object`, then the object is
    returned instead of the ndarray wrapped around it.
    """
    data = numpy.load(path)
    if str(data.dtype) == 'object':
        data = data.item()
    return data


def load_train_input(dataset,
                     normalize=True,
                     normalize_on_the_fly=False):
    """
    Loads the input datapoints for the training set of the dataset named
    `dataset`.

    We suppose the data was created with ifth6266h11/pretraitement/to_npy.py
    that shuffles the train set. So the train set should already by
    shuffled.
    :type dataset: string
    :param dataset: one of the following: 'avicenna', 'harry', 'rita',
                    'sylvester', 'terry'
    :type normalize: Bool
    :param normalize: If True, we normalize the train dataset
                      before returning it
    :type normalize_on_the_fly: Bool
    :param normalize_on_the_fly: If True, we return a Theano Variable that
                                will give as output the normalized value.
                                If the user only take a subtensor of that
                                variable, Theano optimization should make
                                that we will only have in memory the subtensor
                                portion that is computed in normalized form.
                                We store the original data in shared memory
                                in its original dtype.

                                This is usefull to have the original data in
                                its original dtype in memory to same memory.
                                Especialy usefull to be able to use rita and
                                harry with 1G per jobs.
    """
    assert not (normalize and normalize_on_the_fly), \
            "Can't normalize in 2 way at the same time!"
    # TODO: figure out the ule dataset
    assert dataset in ['avicenna',
                       'harry',
                       'rita',
                       'sylvester',
                       'terry']
    fname = os.path.join('/data/lisa/data/UTLC/numpy_data/', dataset +
                         '_train_x.npy')
    data = ift6266h12.load_npy(fname)
    if normalize or normalize_on_the_fly:
        if normalize_on_the_fly:
            # Shared variables of the original type
            data = theano.shared(data,
                                 borrow=True,
                                 name=dataset + "_train")
            # Symbolic variables cast into floatX
            data = theano.tensor.cast(data, theano.config.floatX)
        else:
            data = data.astype(theano.config.floatX)
        if dataset == "ule":
            data /= 255
        elif dataset in ["avicenna", "sylvester"]:
            if dataset == "avicenna":
                data_mean = 514.62154022835455
                data_std = 6.829096494224145
            else:
                data_mean = 403.81889927027686
                data_std = 96.43841050784053
            data -= data_mean
            data /= data_std
        elif dataset == "harry":
            std = 0.69336046033925791  # train.std()slow to compute
            data /= std
        elif dataset == "rita":
            v = numpy.asarray(230, dtype=theano.config.floatX)
            data /= v
        elif dataset == 'terry':
            data = data.astype(theano.config.floatX) / 300
        else:
            raise Exception("This dataset don't have its "
                            "normalization defined")
    return data


def load_train_labels(dataset):
    # TODO: figure out the ule dataset
    assert dataset in ['avicenna',
                       'harry',
                       'rita',
                       'sylvester',
                       'terry']
    fname = os.path.join('/data/lisa/data/UTLC/numpy_data/', dataset +
                         '_train_y.npy')
    data = ift6266h12.load_npy(fname)
    return data


def load_valid_input(dataset,
                     normalize=True,
                     normalize_on_the_fly=False,
                     randomize=False):
    """
    Loads the input datapoints for the training set of the dataset named
    `dataset`.

    We suppose the data was created with ifth6266h11/pretraitement/to_npy.py
    that shuffles the train set. So the train set should already by
    shuffled.
    :type dataset: string
    :param dataset: one of the following: 'avicenna', 'harry', 'rita',
                    'sylvester', 'terry'
    :type normalize: Bool
    :param normalize: If True, we normalize the train dataset
                      before returning it
    :type normalize_on_the_fly: Bool
    :param normalize_on_the_fly: If True, we return a Theano Variable that
                            will give as output the normalized value. If the
                            user only take a subtensor of that variable,
                            Theano optimization should make that we will
                            only have in memory the subtensor portion that
                            is computed in normalized form. We store
                            the original data in shared memory in its
                            original dtype.

                            This is usefull to have the original data in
                            its original dtype in memory to same memory.
                            Especialy usefull to be able to use rita and
                            harry with 1G per jobs.
    :type randomize: Bool
    :param randomize: Change the order of samples
    """
    assert not (normalize and normalize_on_the_fly), \
            "Can't normalize in 2 way at the same time!"
    # TODO: figure out the ule dataset
    assert dataset in ['avicenna',
                       'harry',
                       'rita',
                       'sylvester',
                       'terry']
    fname = os.path.join('/data/lisa/data/UTLC/numpy_data/',
                         dataset + '_valid_x.npy')
    data = ift6266h12.load_npy(fname)
    if randomize:
        rng = numpy.random.RandomState([1, 2, 3, 4])
        perm = rng.permutation(data.shape[0])
        data = data[perm]
    if normalize or normalize_on_the_fly:
        if normalize_on_the_fly:
            # Shared variables of the original type
            data = theano.shared(data,
                                 borrow=True,
                                 name=dataset + "_valid")
            # Symbolic variables cast into floatX
            data = theano.tensor.cast(data, theano.config.floatX)
        else:
            data = data.astype(theano.config.floatX)
        if dataset == "ule":
            data /= 255
        elif dataset in ["avicenna", "sylvester"]:
            if dataset == "avicenna":
                data_mean = 514.62154022835455
                data_std = 6.829096494224145
            else:
                data_mean = 403.81889927027686
                data_std = 96.43841050784053
            data -= data_mean
            data /= data_std
        elif dataset == "harry":
            std = 0.69336046033925791  # train.std()slow to compute
            data /= std
        elif dataset == "rita":
            v = numpy.asarray(230, dtype=theano.config.floatX)
            data /= v
        elif dataset == 'terry':
            data = data.astype(theano.config.floatX) / 300
        else:
            raise Exception("This dataset don't have its "
                            "normalization defined")
    return data


def load_test_input(dataset,
                    normalize=True,
                    normalize_on_the_fly=False,
                    randomize=False):
    """
    Loads the input datapoints for the training set of the dataset named
    `dataset`.

    We suppose the data was created with ifth6266h11/pretraitement/to_npy.py
    that shuffles the train set. So the train set should already by
    shuffled.
    :type dataset: string
    :param dataset: one of the following: 'avicenna', 'harry', 'rita',
                    'sylvester', 'terry'
    :type normalize: Bool
    :param normalize: If True, we normalize the train dataset
                      before returning it
    :type normalize_on_the_fly: Bool
    :param normalize_on_the_fly: If True, we return a Theano Variable that
                            will give as output the normalized value. If the
                            user only take a subtensor of that variable,
                            Theano optimization should make that we will
                            only have in memory the subtensor portion that
                            is computed in normalized form. We store the
                            original data in shared memory in its original
                            dtype.

                            This is usefull to have the original data in its
                            original dtype in memory to same memory. Especialy
                            usefull to be able to use rita and harry with 1G
                            per jobs.
    :type randomize: Bool
    :param randomize: Change the order of samples
    """
    assert not (normalize and normalize_on_the_fly), \
            "Can't normalize in 2 way at the same time!"
    # TODO: figure out the ule dataset
    assert dataset in ['avicenna',
                       'harry',
                       'rita',
                       'sylvester',
                       'terry']
    fname = os.path.join('/data/lisa/data/UTLC/numpy_data/',
                         dataset + '_test_x.npy')
    data = ift6266h12.load_npy(fname)
    if randomize:
        rng = numpy.random.RandomState([1, 2, 3, 4])
        perm = rng.permutation(data.shape[0])
        data = data[perm]
    if normalize or normalize_on_the_fly:
        if normalize_on_the_fly:
            # Shared variables of the original type
            data = theano.shared(data,
                                 borrow=True,
                                 name=dataset + "_test")
            # Symbolic variables cast into floatX
            data = theano.tensor.cast(data, theano.config.floatX)
        else:
            data = data.astype(theano.config.floatX)
        if dataset == "ule":
            data /= 255
        elif dataset in ["avicenna", "sylvester"]:
            if dataset == "avicenna":
                data_mean = 514.62154022835455
                data_std = 6.829096494224145
            else:
                data_mean = 403.81889927027686
                data_std = 96.43841050784053
            data -= data_mean
            data /= data_std
        elif dataset == "harry":
            std = 0.69336046033925791  # train.std()slow to compute
            data /= std
        elif dataset == "rita":
            v = numpy.asarray(230, dtype=theano.config.floatX)
            data /= v
        elif dataset == 'terry':
            data = data.astype(theano.config.floatX) / 300
        else:
            raise Exception("This dataset don't have its "
                            "normalization defined")
    return data
