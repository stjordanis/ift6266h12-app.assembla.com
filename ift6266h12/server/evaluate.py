#!/usr/bin/python
"""
Author : Razvan Pascanu
date: Jan 2012
contact : pascanur [at] iro [dot] umontreal [dot] ca

Code for evaluating computed representations (as done on the UTLC chalange
website).

Usage:
    see : ./evaluate.py -h

Original script from last year ift6266 repo
"""
import numpy
import time
import optparse
import sys
from utils.make_learning_curve import make_learning_curve
from utils.alc import alc
import ift6266h12


def score(dataset,
          labels,
          min_repeat=10,
          max_repeat=500,
          ebar=0.01,
          max_point_num=7,
          debug=False):
    """
    :param dataset: numpy matrix of shape NxD where N is the number of
                examples and D is the number of features
    :param labels: `one-hot` encoded numpy matrix with the target labels
    :param min_repeat: minimum number of repetitions for a specific number
                of examples
    :param max_repeat: maximum number of repetitions for a specific number
                of examples
    :param ebar: error bar for early termination of repetitions in case the
                error is sufficiently low
    :param max_point_num: Maximum number of points on the learning curve
    :param debug: True for printing debugging messanges, False otherwise
    """
    # Make the learning curve
    x, y, e = make_learning_curve(
                dataset,
                labels,
                min_repeat,
                max_repeat,
                ebar,
                max_point_num,
                debug,
                useRPMat=True  # Whether we should use a
                               # precalculated permutation matrix
               )
    # Compute the (normalized) area under the learning curve
    # returns the ALC and the last AUC value
    return alc(x, y)


def get_parser():
    usage = """\
%prog --features=FEAT --dataset=DATASET [options]

This program evaluatest FEAT as features for dataset DATASET."""

    parser = optparse.OptionParser(usage=usage)

    parser.add_option('--features',
                      dest='feat',
                      default=None,
                      help=('npy file (saved using '
                            'numpy.save)'
                            'containing the features'))
    parser.add_option('--dataset',
                      dest='dataset',
                      default=None,
                      help=('For which dataset these features are. '
                            'Pick one of: avicenna, harry, rita, sylvester,'
                            ' terry'))
    parser.add_option('--min-repeat',
                      dest='min_repeat',
                      default=10,
                      help=('minimum number of repetition of training'))
    parser.add_option('--max-repeat',
                      dest='max_repeat',
                      default=500,
                      help=('max number of repetitions of training'))
    parser.add_option('--ebar',
                      dest='ebar',
                      default=.01,
                      help=('Error threshold for early termination of '
                            'repetitions'))
    parser.add_option('--max-point-num',
                      dest='max_point_num',
                      default=7,
                      help=('Maximum number of points on the learning curve'))
    parser.add_option('--debug',
                      dest='debug',
                      default=False,
                      action='store_true',
                      help=('Debug messages'))
    return parser

if __name__ == "__main__":
    parser = get_parser()
    (options, args) = parser.parse_args()
    if options.feat is None:
        parser.print_help()
        sys.exit(0)
    data = ift6266h12.load_npy(options.feat)

    label_files = {}
    label_files['avicenna'] = \
        '/data/lisa/data/UTLC/numpy_data/avicenna_valid_y.npy'
    label_files['harry'] = \
        '/data/lisa/data/UTLC/numpy_data/harry_valid_y.npy'
    label_files['rita'] = \
        '/data/lisa/data/UTLC/numpy_data/rita_valid_y.npy'
    label_files['sylvester'] = \
        '/data/lisa/data/UTLC/numpy_data/sylvester_valid_y.npy'
    label_files['terry'] = \
        '/data/lisa/data/UTLC/numpy_data/terry_valid_y.npy'
    label_data = ift6266h12.load_npy(label_files[options.dataset])

    start = time.clock()
    print '.. computing score'
    print 'data shape', data.shape
    print 'labels shape', label_data.shape
    rval = score(data,
                 label_data,
                 options.min_repeat,
                 options.max_repeat,
                 options.ebar,
                 options.max_point_num,
                 options.debug)
    print 'Score :', rval, \
            ' Computed in %5.2f min' % ((time.clock() - start) / 60.)
