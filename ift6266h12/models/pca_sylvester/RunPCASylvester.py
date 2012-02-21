import numpy
from ift6266h12.utils.ift6266h12_io import load_train_input
from ift6266h12.utils.ift6266h12_io import load_valid_input
from ift6266h12.utils.ift6266h12_io import load_test_input
from sklearn.decomposition import PCA

# Data output location
strOutPath = "/tmp/SylvesterData/"

def RunNormalPCA():
    '''
    This function calculate the first 8 principle components for sylvester dataset
    The number 8 is chosen because it gave the best results for the competition
    '''

    print("Loading training data...")
    data = load_train_input('sylvester' , normalize=False)
    pca = PCA(n_components = 8)
    pca.whiten = True
    print("Applying PCA on training data...")
    pca.fit(data)
    numpy.save(strOutPath + "sylvester_train_pca8.npy", pca.transform(data))
    numpy.save(strOutPath + "components_pca8.npy", pca.components_)


    print("Loading validation data...")
    data = load_valid_input('sylvester' , normalize=False)
    pca = PCA(n_components = 8)
    pca.whiten = True
    print("Applying PCA on validation data...")
    pca.fit(data)
    numpy.save(strOutPath + "sylvester_valid_pca8.npy", pca.transform(data))


    print("Loading testing data...")
    data = load_test_input('sylvester' , normalize=False)
    pca = PCA(n_components = 8)
    pca.whiten = True
    print("Applying PCA on testing data...")
    pca.fit(data)
    numpy.save(strOutPath + "sylvester_test_pca8.npy", pca.transform(data))


def RunTransductivePCA():
    '''
    This is a normal PCA but it is applied on test data. These components will be used as the last layer for phase 1.
    The main idea  to use these components lies beyond the fact that there are new labels in test data that do not exist in training data.
    '''
    print("Loading testing data...")
    data = load_test_input('sylvester' , normalize=False)
    pca = PCA(n_components = 8)
    pca.fit(data)
    numpy.save(strOutPath + "transuctive_components_pca8.npy", pca.components_)
