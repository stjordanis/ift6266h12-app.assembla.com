#!/usr/bin/env python

import os, numpy
from scikits.learn.decomposition import PCA

from ift6266h12.utils.ift6266h12_io import load_train_input, load_test_input, load_valid_input

dest_path = '/data/lisa/data/UTLC/pca'

trainset = load_train_input('sylvester', normalize=True)
testset = load_test_input('sylvester', normalize=True)
validset = load_valid_input('sylvester', normalize=True)

pca = PCA(32)
pca.fit(trainset)

numpy.save(os.path.join(dest_path, 'sylvester_train_x_pca32.npy'), pca.transform(trainset))
numpy.save(os.path.join(dest_path, 'sylvester_valid_x_pca32.npy'), pca.transform(validset))
numpy.save(os.path.join(dest_path, 'sylvester_test_x_pca32.npy'), pca.transform(testset))
