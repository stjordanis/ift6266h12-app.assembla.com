import numpy
import sys
import gzip

filename = sys.argv[1]
f = gzip.open(filename, 'rb')
dt = numpy.loadtxt(f.read())
numpy.save(filename + '_train_x.npy')
