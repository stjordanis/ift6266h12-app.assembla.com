"""A stochastic gradient descent minimizer.
"""
import numpy
import theano

def sgd_updates(params, grads, stepsizes):
    """Return a list of (pairs) that can be used as updates in theano.function to implement
    stochastic gradient descent.

    :param params: variables to adjust in order to minimize some cost
    :type params: a list of variables (theano.function will require shared variables)
    :param grads: the gradient on each param (with respect to some cost)
    :type grads: list of theano expressions
    :param stepsizes: step by this amount times the negative gradient on each iteration
    :type stepsizes: [symbolic] scalar or list of one [symbolic] scalar per param
    """
    try:
        iter(stepsizes)
    except Exception:
        stepsizes = [stepsizes for p in params]
    if len(params) != len(grads):
        raise ValueError('params and grads have different lens')
    updates = [(p, p - step * gp) for (step, p, gp) in zip(stepsizes, params, grads)]
    return updates

def sgd_momentum_updates(params, grads, stepsizes, momentum=0.9):
    # if stepsizes is just a scalar, expand it to match params
    try:
        iter(stepsizes)
    except Exception:
        stepsizes = [stepsizes for p in params]
    try:
        iter(momentum)
    except Exception:
        momentum = [momentum for p in params]
    if len(params) != len(grads):
        raise ValueError('params and grads have different lens')
    headings = [theano.shared(numpy.zeros_like(p.get_value(borrow=True))) for p in params]
    updates = []
    for s, p, gp, m, h in zip(stepsizes, params, grads, momentum, headings):
        updates.append((p, p + s * h))
        updates.append((h, m*h - (1.0-m)*gp))
    return updates


class StochasticGradientDescent(theano.Module):
    """Fixed stepsize gradient descent

    Methods for gradient descent are:
    - step(arg_vals) which returns None and updates the params
    - step_cost(arg_vals) which returns the cost value, and updates the params
    
    """
    def __init__(self, args, cost, params, 
                 gradients=None, stepsize=None, 
                 updates=None, auxout=None, methods=True):
        """
        :param stepsize: the step to take in (negative) gradient direction
        :type stepsize: None, scalar value, or scalar TensorVariable

        :param updates: extra symbolic updates to make when evating either step or step_cost
        (these override the gradients if necessary)
        :type updates: dict Variable -> Variable
        :param auxout: auxiliary outputs, list containing output symbols to 
                      compute at the same time as cost (for efficiency)
        :param methods: Should this module define the step and step_cost methods?
        """
        super(StochasticGradientDescent, self).__init__()
        self.stepsize_init = None

        if stepsize is None:
            self.stepsize = theano.tensor.dscalar()
        elif isinstance(stepsize, theano.tensor.TensorVariable):
            self.stepsize = stepsize
        else:
            self.stepsize = (theano.tensor.as_tensor_variable(stepsize))

        if self.stepsize.ndim != 0:
            raise TypeError('stepsize must be a scalar', stepsize)

        self.params = params
        if gradients is None:
            self.gparams = [theano.tensor.grad(cost, self.params)]
        else:
            self.gparams = gradients
        assert len(self.params) == len(self.gparams)

        self._updates = (dict((p, p - self.stepsize * g) for p, g in zip(self.params, self.gparams)))
        if updates is not None:
            self._updates.update(updates)

        if methods:
            if auxout is None:
                self.step = theano.Method(args, [], updates=self._updates)
                self.step_cost = theano.Method(args, cost, updates=self._updates)
            else:
                # step cost always returns a list if auxout
                self.step = theano.Method(
                        args, [] + auxout,
                        updates=self._updates)
                self.step_cost = theano.Method(
                        args, [cost]+auxout,
                        updates=self._updates)


    updates = property(lambda self: self._updates.copy())

    def _instance_initialize(self, obj):
        pass

def sgd_minimizer(stepsize=None):
    """Curry the stepsize argument to StochasticGradientDescent, providing standard minimizer interface
    
    :returns: standard minimizer constructor f(args, cost, params, gradient=None)
    """
    def f(args, cost, params, gradients=None, updates=None, auxout=None):
        return StochasticGradientDescent(args, cost, params, gradients=gradients, stepsize=stepsize,
                updates=updates, auxout=auxout)
    return f
