import functools
import numpy


""" Collection for preprocessing tools"""


def nonzero_features(data, combine=None):
    """
    Get features for which there are nonzero entries in the data.

    Note: I would return a mask (bool array) here, but scipy.sparse doesn't
    appear to fully support advanced indexing.

    Parameters
    ----------
    data : list of matrices
        List of data matrices, either in sparse format or not.
        They must have the same number of features (column number).
    combine : function
        A function to combine elementwise which features to keep
        Default keeps the intersection of each non-zero columns

    Returns
    -------
    indices : ndarray object
        Indices of the nonzero features.
    """

    if combine is None:
        combine = functools.partial(reduce, numpy.logical_and)

    # Assumes all values are >0, which is the case for all sparse datasets.
    masks = numpy.asarray([subset.sum(axis=0) for subset in data]).squeeze()
    nz_feats = combine(masks).nonzero()[0]

    return nz_feats


def filter_nonzero(data, combine=None):
    """
    Filter non-zero features of data according to a certain combining function

    Parameters
    ----------
    data : list of matrices
        List of data matrices, either in sparse format or not.
        They must have the same number of features (column number).
    combine : function
        A function to combine elementwise which features to keep
        Default keeps the intersection of each non-zero columns
    """

    nz_feats = nonzero_features(data, combine)
    return [set[:, nz_feats] for set in data]


