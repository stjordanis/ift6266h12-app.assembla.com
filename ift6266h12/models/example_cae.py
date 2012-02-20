"""An example of how to use the library so far."""
# Standard library imports
import sys

# Third-party imports
import numpy
import theano
from theano import tensor

try:
    import pylearn2
except ImportError:
    print >>sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

# Local imports
from pylearn2.cost import SquaredError
from pylearn2.autoencoder import ContractiveAutoencoder, build_stacked_ae
# TODO use new pylearn optimzer. Need some changes though
from optimizer import SGDOptimizer

if __name__ == "__main__":
    # Simulate some fake data.
    rng = numpy.random.RandomState(seed=42)
    data = rng.normal(size=(1000, 15))

    conf = {
        'corruption_level': 0.1,
        'nhid': 20,
        'nvis': data.shape[1],
        'anneal_start': 100,
        'base_lr': 0.01,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': None,
        #'lr_hb': 0.10,
        #'lr_vb': 0.10,
        'irange': 0.001,
    }

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()
    minibatch = theano.printing.Print('min')(minibatch)

    # Allocate a denoising autoencoder with binomial noise corruption.
    cae = ContractiveAutoencoder(conf['nvis'], conf['nhid'],
                                 conf['act_enc'], conf['act_dec'])

    # Allocate an optimizer, which tells us how to update our model.
    cost = SquaredError(cae)(minibatch, cae.reconstruct(minibatch)).mean()
    cost += cae.contraction_penalty(minibatch).mean()
    trainer = SGDOptimizer(cae, conf['base_lr'], conf['anneal_start'])
    updates = trainer.cost_updates(cost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch], cost, updates=updates, allow_input_downcast=True)

    # Suppose we want minibatches of size 10
    batchsize = 10

    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    for epoch in xrange(5):
        for offset in xrange(0, data.shape[0], batchsize):
            minibatch_err = train_fn(data[offset:(offset + batchsize)])
            print ("epoch %d, batch %d-%d: %f" %
                   (epoch, offset, offset + batchsize - 1, minibatch_err))

    # Suppose you then want to use the representation for something.
    transform = theano.function([minibatch], cae(minibatch), allow_input_downcast=True)

    print "Transformed data:"
    print numpy.histogram(transform(data))
