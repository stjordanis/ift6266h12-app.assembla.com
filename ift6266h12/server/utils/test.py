import numpy as N
import make_learning_curve


def test(model, data, verbose=False):
    if verbose:
        print '==> Testing hebbian... '
    #

    p, n = data.X.shape

    assert n == model.W.shape[0]

    Yest = N.dot(data.X, model.W.T) + model.b0
    # Remove ties (the negative class is usually most abundant)
    Yest[Yest == 0.] = -1e-12
    rdata = make_learning_curve.data_struct(Yest, data.Y)

    if verbose:
        print 'done'
    return rdata
