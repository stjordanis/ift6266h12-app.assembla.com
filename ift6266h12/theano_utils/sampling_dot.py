"""
This is an op implemented by Yann Dauphin that for Theano, but which was not
reviewed yet and not integrated in Theano. Because is needed in the calss,
I provide a copy of it through this repository.

I did not review the code, and I leave this to Theano developers team.
Razvan Feb 23, 2012
"""

import sys
import numpy
import theano
import scipy.sparse
from theano.printing import Print


from theano import gof
from theano import tensor
from theano import compile
from theano import scalar
from theano import config
from theano import gradient
from theano.tensor import blas
import logging
# So that we have same things available in this file
from theano.sparse.basic import *

_dot = theano.sparse.basic.Dot()

_mtype_to_str = {scipy.sparse.csc_matrix: "csc",
                 scipy.sparse.csr_matrix: "csr"}


def _is_sparse_variable(x):
    """
    @rtype: boolean
    @return: True iff x is a L{SparseVariable} (and not a L{tensor.TensorType})
    """
    if not isinstance(x.type, (SparseType, tensor.TensorType)):
        raise NotImplementedError("this function should only be called on "
                                  "*variables* (of type sparse.SparseType "
                                  "or tensor.TensorType), not,", x)
    return isinstance(x.type, SparseType)


def _is_dense_variable(x):
    """
    @rtype: boolean
    @return: True unless x is a L{SparseVariable} (and not a
    L{tensor.TensorType})
    """
    if not isinstance(x.type, (SparseType, tensor.TensorType)):
        raise NotImplementedError("this function should only be called on "
                                  "*variables* (of type sparse.SparseType or "
                                  "tensor.TensorType), not,", x)
    return isinstance(x.type, tensor.TensorType)


def _is_sparse(x):
    """
    @rtype: boolean
    @return: True iff x is a L{scipy.sparse.spmatrix} (and not a
    L{numpy.ndarray})
    """
    if not isinstance(x, (scipy.sparse.spmatrix, numpy.ndarray)):
        raise NotImplementedError("this function should only be called on "
                                  "sparse.scipy.sparse.spmatrix or "
                                  "numpy.ndarray, not,", x)
    return isinstance(x, scipy.sparse.spmatrix)


def _is_dense(x):
    """
    @rtype: boolean
    @return: True unless x is a L{scipy.sparse.spmatrix} (and not a
    L{numpy.ndarray})
    """
    if not isinstance(x, (scipy.sparse.spmatrix, numpy.ndarray)):
        raise NotImplementedError("this function should only be called on "
                                  "sparse.scipy.sparse.spmatrix or "
                                  "numpy.ndarray, not,", x)
    return isinstance(x, numpy.ndarray)


def _kmap_eq(a, b):
    if a is None and b is None:
        return True
    return numpy.all(a == b)


def _kmap_hash(a):
    if a is None:
        return 12345
    return hash(numpy.str(a))

class SamplingDot(gof.op.Op):
    """
    Operand for calculating the dot product DOT(X, Y) = Z when you only want
    to calculate a subset of Z. It is equivalent to P o (X . Y) where o is
    the element-wise product, X and Y operands of the dot product and P is
    a matrix that contains 1 when the corresponding element of Z should be
    calculated and 0 when it shouldn't. Note that SamplingDot has a different
    interface than DOT because SamplingDot requires X to be a MxK matrix
    while Y is a NxK matrix instead of the usual KxN matrix.

    It will work if the pattern is not binary value, but if the pattern
    doesn't have a high sparsity proportion it will be slower then a more
    optimized dot followed by a normal elemwise multiplication.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'SamplingDot'

    def make_node(self, x, y, pattern):
        if (_is_sparse_variable(x) or
            _is_sparse_variable(y) or
            _is_sparse_variable(pattern)):
            raise TypeError(x)

        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        pattern = tensor.as_tensor_variable(pattern)

        dtype_out = scalar.upcast(x.type.dtype,
                                  y.type.dtype,
                                  pattern.type.dtype)

        return gof.Apply(self,
                         [x, y, pattern],
                         [tensor.tensor(dtype=dtype_out,
                                        broadcastable=(False, False))])

    def perform(self, node, (x, y, pattern), (out, )):
        if (_is_sparse_variable(x) or
            _is_sparse_variable(y) or
            _is_sparse_variable(y)):
            raise TypeError(x)

        rval = pattern * numpy.dot(x, y.T)

        out[0] = rval

    def grad(self, (x, y, pattern), (gz,)):
        rval = [
            sampling_dot_grad(gz, y, pattern),
            sampling_dot_grad(gz.T, x, pattern.T),
            None
        ]

        return rval
sampling_dot = SamplingDot()


class SamplingDotGrad(gof.op.Op):
    """
    Gradient of the SamplingDot operation.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'SamplingDotGrad'

    def make_node(self, x, y, pattern):
        if (_is_sparse_variable(x) or
            _is_sparse_variable(y) or
            _is_sparse_variable(pattern)):
            raise TypeError(x)

        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        pattern = tensor.as_tensor_variable(pattern)

        dtype_out = scalar.upcast(x.type.dtype,
                                  y.type.dtype,
                                  pattern.type.dtype)

        return gof.Apply(self,
                         [x, y, pattern],
                         [tensor.tensor(dtype=dtype_out,
                                        broadcastable=(False, False))])

    def perform(self, node, (x, y, pattern), (out, )):
        if (_is_sparse_variable(x) or
            _is_sparse_variable(y) or
            _is_sparse_variable(y)):
            raise TypeError(x)

        rval = numpy.dot(pattern * x, y)

        out[0] = rval
sampling_dot_grad = SamplingDotGrad()


# This optimization is useful for networks with tied weights
local_tied = gof.opt.PatternSub(
    (tensor.sub,
     'z',
     (tensor.mul,
      {'pattern': 'alpha',
       'constraint': lambda expr: numpy.all(expr.type.broadcastable)},
      (tensor.add,
       (sampling_dot_grad, 'x0', 'y0', 'pattern'),
       (_dot, 'x1', 'y1')))),
    (tensor.sub,
     (tensor.sub,
      'z',
      (tensor.mul, 'alpha', (_dot, 'x1', 'y1'))),
     (tensor.mul, 'alpha', (sampling_dot_grad, 'x0', 'y0', 'pattern'))))
register_specialize(local_tied, name="local_tied")


class SamplingDotDense(gof.Op):
    """
    Optimized SamplingDot when the pattern P is a dense matrix.

    If we have the input of mixed dtype, we insert cast elemwise in the graph
    to be able to call blas function as they don't allow mixed dtype.

    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'SamplingDot{Dense}'

    def make_node(self, x, y, pattern):
        if (_is_sparse_variable(x) or
            _is_sparse_variable(y) or
            _is_sparse_variable(pattern)):
            raise TypeError(x)

        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        pattern = tensor.as_tensor_variable(pattern)

        dtype_out = scalar.upcast(x.type.dtype,
                                  y.type.dtype,
                                  pattern.type.dtype)
        dot_out = scalar.upcast(x.type.dtype, y.type.dtype)

        # We call blas ?dot function that take only param of the same type
        x = tensor.cast(x, dot_out)
        y = tensor.cast(y, dot_out)

        return gof.Apply(self,
                         [x, y, pattern],
                         [tensor.tensor(dtype=dtype_out,
                                        broadcastable=(False, False))])

    #def perform(self, node, (x, y, pattern), (out,)):
    #    raise NotImplemented()

    def c_support_code(self):
        return blas.blas_header_text()

    def c_libraries(self):
        return blas.ldflags()

    def c_compile_args(self):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return blas.ldflags(libs=False, include_dir=True)

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, (x, y, pattern), (z,), sub):
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for x')
        if node.inputs[1].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for y')
        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError(
                'Complex types are not supported for pattern')

        dot_out = scalar.upcast(node.inputs[0].type.dtype,
                                node.inputs[0].type.dtype)

        if dot_out == "float32":
            conv_type = "float"
            cdot = "sdot_"
        else:
            conv_type = "double"
            cdot = "ddot_"
        # retrieve dtype number
        typenum_x = node.inputs[0].type.dtype_specs()[-1]
        # retrieve dtype number
        typenum_y = node.inputs[1].type.dtype_specs()[-1]
        # retrieve dtype number
        typenum_pattern = node.inputs[2].type.dtype_specs()[-1]
        # retrieve dtype number
        typenum_z = tensor.TensorType(
            node.outputs[0].dtype, []).dtype_specs()[-1]
        rval = """
        if (%(x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(pattern)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(pattern) != 2"); %(fail)s;}

        if (%(x)s->descr->type_num != %(typenum_x)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x"); %(fail)s;}

        if (%(y)s->descr->type_num != %(typenum_y)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); %(fail)s;}

        if (%(pattern)s->descr->type_num != %(typenum_pattern)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for pattern"); %(fail)s;}

        if (%(x)s->dimensions[1] != %(y)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows"); %(fail)s;}

        if (%(pattern)s->dimensions[0] != %(x)s->dimensions[0] || %(pattern)s->dimensions[1] != %(y)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of the pattern and the output must match"); %(fail)s;}

        if (!%(z)s
            || (%(z)s->dimensions[0] != %(x)s->dimensions[0])
            || (%(z)s->dimensions[1] != %(y)s->dimensions[0])
            || (%(z)s->descr->type_num != %(typenum_z)s)
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0,0};
            dims[0] = %(x)s->dimensions[0];
            dims[1] = %(y)s->dimensions[0];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = %(z)s->dimensions[0];
            npy_intp N = %(z)s->dimensions[1];
            npy_intp K = %(y)s->dimensions[1];

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s* __restrict__ Dz   = (dtype_%(z)s*)%(z)s->data;
            const dtype_%(x)s* __restrict__ Dx = (dtype_%(x)s*)%(x)s->data;
            const dtype_%(y)s* __restrict__ Dy = (dtype_%(y)s*)%(y)s->data;
            const dtype_%(pattern)s* __restrict__ Dpattern = (dtype_%(pattern)s*)%(pattern)s->data;

            const npy_intp Sdz = %(z)s->strides[1]/%(z)s->descr->elsize;
            const npy_intp Sdx = %(x)s->strides[1]/%(x)s->descr->elsize;
            const npy_intp Sdy = %(y)s->strides[1]/%(y)s->descr->elsize;
            const npy_intp Sdp = %(pattern)s->strides[1]/%(pattern)s->descr->elsize;

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_%(z)s));

            for (npy_int32 m = 0; m < M; ++m) {
                // pointer to m-th row of the output matrix Z
                dtype_%(z)s* const __restrict__ zm = (dtype_%(z)s*)(%(z)s->data + %(z)s->strides[0] * m);

                const dtype_%(pattern)s* p_row = (dtype_%(pattern)s*)(%(pattern)s->data + %(pattern)s->strides[0] * m);

                for (npy_int32 n = 0; n < N; ++n) {
                    if (*(p_row + n * Sdp) != 0) {
                        const dtype_%(x)s* x_row = (dtype_%(x)s*)(%(x)s->data + %(x)s->strides[0] * m);

                        const dtype_%(y)s* y_col = (dtype_%(y)s*)(%(y)s->data + %(y)s->strides[0] * n);
                        *(zm + n * Sdz) = %(cdot)s((int*)&K, (const %(conv_type)s*)x_row, (int*)&Sdx, (const %(conv_type)s*)y_col, (int*)&Sdy) * *(p_row + n * Sdp);
                    }
                }
            }

        }
        """ % dict(locals(), **sub)

        return rval
sampling_dot_dense = SamplingDotDense()

local_sampling_dot_dense = gof.opt.PatternSub((sampling_dot, 'x', 'y', 'pattern'),
                                              (sampling_dot_dense, 'x', 'y', 'pattern'),
                                              name='local_sampling_dot_dense')
register_specialize(local_sampling_dot_dense, name="local_sampling_dot_dense")


class SamplingDotDenseGrad(gof.Op):
    """
    Optimized gradient of SamplingDot when the pattern P is a dense matrix.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'SamplingDotGrad{Dense}'

    def make_node(self, x, y, pattern):
        if (_is_sparse_variable(x) or
            _is_sparse_variable(y) or
            _is_sparse_variable(pattern)):
            raise TypeError(x)

        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        pattern = tensor.as_tensor_variable(pattern)

        dtype_out = scalar.upcast(x.type.dtype, y.type.dtype, pattern.type.dtype)
        blas_out = scalar.upcast(x.type.dtype, y.type.dtype)

        # We call blas ?axpy function that take only param of the same type
        x = tensor.cast(x, blas_out)
        y = tensor.cast(y, blas_out)

        return gof.Apply(self, [x, y, pattern], [tensor.tensor(dtype=dtype_out, broadcastable=(False, False))])

    #def perform(self, node, (x, y, pattern), (out,)):
    #    raise NotImplemented()

    def c_support_code(self):
        return blas.blas_header_text()

    def c_libraries(self):
        return blas.ldflags()

    def c_compile_args(self):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return blas.ldflags(libs=False, include_dir=True)

    def c_code(self, node, name, (x, y, pattern), (z,), sub):
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for x')
        if node.inputs[1].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for y')
        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for pattern')

        blas_out = scalar.upcast(node.inputs[0].type.dtype, node.inputs[0].type.dtype)
        if blas_out == "float32":
            conv_type = "float"
            axpy = "saxpy_"
        else:
            conv_type = "double"
            axpy = "daxpy_"

        typenum_x = node.inputs[0].type.dtype_specs()[-1] # retrieve dtype number
        typenum_y = node.inputs[1].type.dtype_specs()[-1] # retrieve dtype number
        typenum_pattern = node.inputs[2].type.dtype_specs()[-1] # retrieve dtype number
        typenum_z = tensor.TensorType(node.outputs[0].dtype, []).dtype_specs()[-1] # retrieve dtype number

        rval = """
        if (%(x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(pattern)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(pattern) != 2"); %(fail)s;}

        if (%(x)s->descr->type_num != %(typenum_x)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x"); %(fail)s;}

        if (%(y)s->descr->type_num != %(typenum_y)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); %(fail)s;}

        if (%(pattern)s->descr->type_num != %(typenum_pattern)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for pattern"); %(fail)s;}

        if (%(x)s->dimensions[1] != %(y)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows"); %(fail)s;}

        if (%(pattern)s->dimensions[0] != %(x)s->dimensions[0] || %(pattern)s->dimensions[1] != %(x)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of x and the pattern must match"); %(fail)s;}

        if (!%(z)s
            || (%(z)s->dimensions[0] != %(x)s->dimensions[0])
            || (%(z)s->dimensions[1] != %(y)s->dimensions[1])
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0,0};
            dims[0] = %(x)s->dimensions[0];
            dims[1] = %(y)s->dimensions[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = %(z)s->dimensions[0];
            npy_intp N = %(z)s->dimensions[1];
            npy_intp K = %(y)s->dimensions[0];

            // pointers to access actual data in the arrays passed as params.
            dtype_%(z)s* __restrict__ Dz   = (dtype_%(z)s*)%(z)s->data;
            const dtype_%(x)s* __restrict__ Dx = (dtype_%(x)s*)%(x)s->data;
            const dtype_%(y)s* __restrict__ Dy = (dtype_%(y)s*)%(y)s->data;
            const dtype_%(pattern)s* __restrict__ Dpattern = (dtype_%(pattern)s*)%(pattern)s->data;

            const npy_intp Sdz = %(z)s->strides[1]/%(z)s->descr->elsize;
            const npy_intp Sdx = %(x)s->strides[1]/%(x)s->descr->elsize;
            const npy_intp Sdy = %(y)s->strides[1]/%(y)s->descr->elsize;
            const npy_intp Sdp = %(pattern)s->strides[1]/%(pattern)s->descr->elsize;

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_%(z)s));

            for (npy_int32 m = 0; m < M; ++m) {
                // pointer to m-th row of the output matrix Z
                dtype_%(z)s* const __restrict__ z_row = (dtype_%(z)s*)(%(z)s->data + %(z)s->strides[0] * m);

                const dtype_%(pattern)s* p_row = (dtype_%(pattern)s*)(%(pattern)s->data + %(pattern)s->strides[0] * m);

                const dtype_%(x)s* x_row = (dtype_%(x)s*)(%(x)s->data + %(x)s->strides[0] * m);

                for (npy_int32 k = 0; k < K; ++k) {
                    if (*(p_row + k*Sdp) != 0) {
                        const dtype_%(x)s* y_row = (dtype_%(y)s*)(%(y)s->data + %(y)s->strides[0] * k);

                        %(axpy)s((int*)&N, (%(conv_type)s*)(x_row + k*Sdx), (%(conv_type)s*)y_row, (int*)&Sdy, (%(conv_type)s*)z_row, (int*)&Sdz);
                    }
                }
            }

        }
        """% dict(locals(), **sub)

        return rval
sampling_dot_dense_grad = SamplingDotDenseGrad()

local_sampling_dot_dense_grad = gof.opt.PatternSub((sampling_dot_grad, 'x', 'y', 'pattern'),
                                                     (sampling_dot_dense_grad, 'x', 'y', 'pattern'))
register_specialize(local_sampling_dot_dense_grad, name="local_sampling_dot_dense_grad")


class SamplingDotDenseGradUpdate(gof.Op):
    """
    Optimized gradient of SamplingDot when the pattern P is a dense matrix and
    operations can be performed in-place on the result.

    this perform this pattern of computation: z = z + alpha * sampling_dot_dense_grad(x,y,pattern)
    when the flag inplace is set.

    where z,x,y,pattern are 2d tensor
    and alpha is a scalar.
    """
    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map={ 0 : [0] }

    def __str__(self):
        if self.inplace:
            return 'SamplingDotGradUpdate{Dense}{inplace}'
        else:
            return 'SamplingDotGradUpdate{Dense}{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)) and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self))

    def make_node(self, z, a, x, y, pattern):
        z = tensor.as_tensor_variable(z)
        a = tensor.as_tensor_variable(a)
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        pattern = tensor.as_tensor_variable(pattern)

        assert z.ndim == a.ndim == x.ndim == y.ndim == pattern.ndim == 2
        assert a.type.broadcastable == (True, True)

        if x.type.dtype != y.type.dtype != z.type.dtype != a.type.dtype:
            raise TypeError(x)

        if _is_sparse_variable(x) or _is_sparse_variable(y) or _is_sparse_variable(pattern) or _is_sparse_variable(z):
            raise TypeError(x)

        dtype_out = scalar.upcast(z.type.dtype, a.type.dtype, x.type.dtype, y.type.dtype, pattern.type.dtype)

        # We call blas ?axpy function that take only param of the same type
        z = tensor.cast(z, dtype_out)
        a = tensor.cast(a, dtype_out)
        x = tensor.cast(x, dtype_out)
        y = tensor.cast(y, dtype_out)
        pattern = tensor.cast(pattern, dtype_out)

        if self.inplace:
            assert z.type.dtype == dtype_out

        return gof.Apply(self, [z, a, x, y, pattern], [tensor.tensor(dtype=dtype_out, broadcastable=(False, False))])

    #def perform(self, node, (z, a, x, y, pattern), (out,)):
    #    raise NotImplemented()

    def c_support_code(self):
        return blas.blas_header_text()

    def c_libraries(self):
        return blas.ldflags()

    def c_compile_args(self):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return blas.ldflags(libs=False, include_dir=True)

    def c_code(self, node, name, (z, a, x, y, pattern), (out,), sub):
        if node.inputs[0].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for z')
        if node.inputs[1].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for a')
        if node.inputs[2].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for x')
        if node.inputs[3].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for y')
        if node.inputs[4].type.dtype in ('complex64', 'complex128'):
            raise NotImplementedError('Complex types are not supported for pattern')

        if node.outputs[0].type.dtype == "float32":
            conv_type = "float"
            axpy = "saxpy_"
        else:
            conv_type = "double"
            axpy = "daxpy_"

        typenum_z = node.inputs[0].type.dtype_specs()[-1] # retrieve dtype number
        typenum_a = node.inputs[1].type.dtype_specs()[-1] # retrieve dtype number
        typenum_x = node.inputs[2].type.dtype_specs()[-1] # retrieve dtype number
        typenum_y = node.inputs[3].type.dtype_specs()[-1] # retrieve dtype number
        typenum_pattern = node.inputs[4].type.dtype_specs()[-1] # retrieve dtype number
        typenum_z = tensor.TensorType(node.outputs[0].type.dtype, []).dtype_specs()[-1] # retrieve dtype number

        inplace = int(self.inplace)

        rval = """
        if (%(z)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}
        if (%(x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(pattern)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(pattern) != 2"); %(fail)s;}

        if (%(x)s->descr->type_num != %(typenum_x)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x"); %(fail)s;}

        if (%(y)s->descr->type_num != %(typenum_y)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); %(fail)s;}

        if (%(z)s->descr->type_num != %(typenum_z)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for z"); %(fail)s;}

        if (%(pattern)s->descr->type_num != %(typenum_pattern)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for pattern"); %(fail)s;}

        if (%(a)s->descr->type_num != %(typenum_a)s) {
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for pattern"); %(fail)s;}

        if (%(x)s->dimensions[1] != %(y)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows"); %(fail)s;}

        if (%(z)s->dimensions[0] != %(x)s->dimensions[0] || %(z)s->dimensions[1] != %(y)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of z and the output must match"); %(fail)s;}

        if (%(pattern)s->dimensions[0] != %(x)s->dimensions[0] || %(pattern)s->dimensions[1] != %(x)s->dimensions[1])
        {PyErr_SetString(PyExc_NotImplementedError, "The dimension of x and the pattern must match"); %(fail)s;}

        if (PyArray_SIZE(%(a)s) != 1)
        {PyErr_SetString(PyExc_NotImplementedError, "The number of element in a must be 1"); %(fail)s;}

        if (%(inplace)s)
        {
            Py_XDECREF(%(out)s);
            %(out)s = %(z)s;
            Py_INCREF(%(out)s);
        }
        else if (!%(out)s
            || (%(out)s->dimensions[0] != %(x)s->dimensions[0])
            || (%(out)s->dimensions[1] != %(y)s->dimensions[1])
            )
        {
            {Py_XDECREF(%(out)s);}
            npy_intp dims[] = {0,0};
            dims[0] = %(x)s->dimensions[0];
            dims[1] = %(y)s->dimensions[1];
            %(out)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, %(typenum_z)s);
        }

        {
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = %(out)s->dimensions[0];
            npy_intp N = %(out)s->dimensions[1];
            npy_intp K = %(y)s->dimensions[0];

            // pointers to access actual data in the arrays passed as params.
            dtype_%(out)s* __restrict__ Do   = (dtype_%(out)s*)%(out)s->data;
            dtype_%(z)s* __restrict__ Dz   = (dtype_%(z)s*)%(z)s->data;
            const dtype_%(x)s* __restrict__ Dx = (dtype_%(x)s*)%(x)s->data;
            const dtype_%(y)s* __restrict__ Dy = (dtype_%(y)s*)%(y)s->data;
            const dtype_%(pattern)s* __restrict__ Dpattern = (dtype_%(pattern)s*)%(pattern)s->data;
            const dtype_%(a)s a = ((dtype_%(a)s*)%(a)s->data)[0];

            const npy_intp Sdz = %(z)s->strides[1]/%(z)s->descr->elsize;
            const npy_intp Sdx = %(x)s->strides[1]/%(x)s->descr->elsize;
            const npy_intp Sdy = %(y)s->strides[1]/%(y)s->descr->elsize;
            const npy_intp Sdp = %(pattern)s->strides[1]/%(pattern)s->descr->elsize;

            if (!(%(inplace)s))
            {
                memcpy(Do, Dz, M*N*sizeof(dtype_%(out)s));
            }

            for (npy_int32 m = 0; m < M; ++m) {
                // pointer to m-th row of the output matrix Z
                dtype_%(out)s* const __restrict__ out_row = (dtype_%(out)s*)(%(out)s->data + %(out)s->strides[0] * m);

                const dtype_%(pattern)s* p_row = (dtype_%(pattern)s*)(%(pattern)s->data + %(pattern)s->strides[0] * m);

                const dtype_%(x)s* x_row = (dtype_%(x)s*)(%(x)s->data + %(x)s->strides[0] * m);

                for (npy_int32 k = 0; k < K; ++k) {
                    if (*(p_row + k*Sdp) != 0) {
                        const dtype_%(x)s* y_row = (dtype_%(y)s*)(%(y)s->data + %(y)s->strides[0] * k);

                        const dtype_%(x)s vx = a * (*(x_row + k*Sdx));

                        %(axpy)s((int*)&N, (%(conv_type)s*)&vx, (%(conv_type)s*)y_row, (int*)&Sdy, (%(conv_type)s*)out_row, (int*)&Sdz);
                    }
                }
            }

        }
        """% dict(locals(), **sub)

        return rval
sampling_dot_dense_grad_update = SamplingDotDenseGradUpdate(False)
sampling_dot_dense_grad_update_inplace = SamplingDotDenseGradUpdate(True)

local_sddgu = gof.opt.PatternSub((tensor.sub, 'z', (tensor.mul, {'pattern' : 'alpha', 'constraint' : lambda expr: numpy.all(expr.type.broadcastable) },
                                                                (sampling_dot_dense_grad, 'x', 'y', 'pattern'))),
                                 (sampling_dot_dense_grad_update, 'z', (tensor.neg, 'alpha'), 'x', 'y', 'pattern'))
register_specialize(local_sddgu, name="local_sddgu")

@gof.local_optimizer([sampling_dot_dense_grad_update])
def local_sddgu_inplace(node):
    if (node.op == sampling_dot_dense_grad_update
        and node.inputs[0].type == node.outputs[0].type):
        return [sampling_dot_dense_grad_update_inplace(*node.inputs)]
register_specialize(local_sddgu_inplace, 'inplace')



