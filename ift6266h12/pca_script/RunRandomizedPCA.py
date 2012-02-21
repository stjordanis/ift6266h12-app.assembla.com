# -*- coding: utf-8 -*-


import numpy
from ift6266h12.utils.ift6266h12_io import load_train_input
from ift6266h12.utils.ift6266h12_io import load_valid_input
from ift6266h12.utils.ift6266h12_io import load_test_input
from sklearn.decomposition import RandomizedPCA

def RunRandomizedPCA():

    # Data output location
    strOutPath = "../Data/numpy_data/"


    # Import data
    print("Loading data")
    data_train = load_train_input('terry', normalize=True)
    data_valid = load_valid_input('terry', normalize=True)
    data_test = load_test_input('terry', normalize=True)

    # Print shape of input data
    print(data_train.shape)

    # Initialize PCA using 10000 components on Terry, twice as much as last year's data
    pca = RandomizedPCA(n_components = 800)

    # Compute PCA
    print("Running pca")
    pca.fit(data_train)
    print("PCA calculation finished.  Printing explaned variance ration for last components")
    print '{0:5f}%'.format(numpy.sum(pca.explained_variance_ratio_)*100)

    # Apply transform
    print("Applying transform")
    data_train_transformed = pca.transform(data_train)
    data_valid_transformed = pca.transform(data_valid)
    data_test_transformed = pca.transform(data_test)

    # Save data
    print("Saving data...")
    numpy.save(strOutPath + "terry_train_x_pca.npy", data_train_transformed)
    numpy.save(strOutPath + "terry_valid_x_pca.npy", data_valid_transformed)
    numpy.save(strOutPath + "terry_test_x_pca.npy", data_test_transformed)
    print("Saving data...  Done")


if __name__ == '__main__':
    RunRandomizedPCA()


