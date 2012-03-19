# imports
import time

from pylearn2.scripts.train import Train
from pylearn2.scripts.train import FeatureDump
from pylearn2.datasets.npy_npz import NpyDataset
from pylearn2.corruption import BinomialCorruptor
from pylearn2.autoencoder import DenoisingAutoencoder
from pylearn2.training_algorithms.sgd import UnsupervisedExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from pylearn2.utils.serial import load

from ift6266h12.server.evaluate import score
import ift6266h12.utils.ift6266h12_io as ift6266h12

from scikits.learn.decomposition import PCA
import numpy as np

def main_train(
               work_dir = "../results/avicenna/",
               corruption_level = 0.3,
               nvis = 75,
               nhid = 600,
               tied_weights = True,
               act_enc = "sigmoid",
               act_dec = None,
               max_epochs = 2,
               learning_rate = 0.001,
               batch_size = 20,
               monitoring_batches = 5,
               save_freq = 1,
               n_components_trans_pca = 7
               ):
    
    conf = {
        'corruption_level' : corruption_level,
        'nvis' : nvis,
        'nhid' : nhid,
        'tied_weights' : tied_weights,
        'act_enc' : act_enc,
        'act_dec' : act_dec,
        'max_epochs' : max_epochs,
        'learning_rate' : learning_rate,
        'batch_size' : batch_size,
        'monitoring_batches' : monitoring_batches,
        'save_freq' : save_freq,
        'n_components_trans_pca' : n_components_trans_pca
    }
            
    start = time.clock()
    
    ###############   TRAIN THE DAE
    train_file = work_dir + "train_pca" + str(conf['nvis']) + ".npy"
    save_path = work_dir + "train_pca" + str(conf['nvis']) + "_dae" + str(conf['nhid']) + "_model.pkl"
    
    trainset = NpyDataset(file = train_file)
    trainset.yaml_src = 'script'
    corruptor = BinomialCorruptor(corruption_level = conf['corruption_level'])
    dae = DenoisingAutoencoder(nvis = conf['nvis'], nhid = conf['nhid'], tied_weights = conf['tied_weights'], corruptor = corruptor, act_enc = conf['act_enc'], act_dec = conf['act_dec'])
    cost = MeanSquaredReconstructionError()
    termination_criterion = EpochCounter(max_epochs = conf['max_epochs'])
    algorithm = UnsupervisedExhaustiveSGD(learning_rate = conf['learning_rate'], batch_size = conf['batch_size'], monitoring_batches = conf['monitoring_batches'], monitoring_dataset = trainset, cost = cost, termination_criterion = termination_criterion)
    
    train_obj = Train(dataset = trainset, model = dae, algorithm = algorithm, save_freq = conf['save_freq'], save_path = save_path)
    train_obj.main_loop()

        
    ###############   APPLY THE MODEL ON THE TRAIN DATASET
    print("Applying the model on the train dataset...")
    model = load(save_path)
    save_train_path = work_dir + "train_pca" + str(conf['nvis']) + "_dae" + str(conf['nhid']) + ".npy"
    dump_obj = FeatureDump(encoder = model, dataset = trainset, path = save_train_path)
    dump_obj.main_loop()
           
            
    ###############   APPLY THE MODEL ON THE VALID DATASET
    print("Applying the model on the valid dataset...")    
    valid_file = work_dir + "valid_pca" + str(conf['nvis']) + ".npy"

    validset = NpyDataset(file = valid_file)
    validset.yaml_src = 'script'
    save_valid_path = work_dir + "valid_pca" + str(conf['nvis']) + "_dae" + str(conf['nhid']) + ".npy"
    dump_obj = FeatureDump(encoder = model, dataset = validset, path = save_valid_path)
    dump_obj.main_loop()

            
    ###############   APPLY THE MODEL ON THE TEST DATASET
    print("Applying the model on the test dataset...") 
    test_file = work_dir + "test_pca" + str(conf['nvis']) + ".npy"
            
    testset = NpyDataset(file = test_file)
    testset.yaml_src = 'script'
    save_test_path = work_dir + "test_pca" + str(conf['nvis']) + "_dae" + str(conf['nhid']) + ".npy"
    dump_obj = FeatureDump(encoder = model, dataset = testset, path = save_test_path)
    dump_obj.main_loop()
    
            
    ###############   COMPUTE THE ALC SCORE ON VALIDATION SET
    valid_data = ift6266h12.load_npy(save_valid_path)
    label_data = ift6266h12.load_npy('/data/lisa/data/UTLC/numpy_data/avicenna_valid_y.npy')
    alc_1 = score(valid_data, label_data)

            
    ###############   APPLY THE TRANSDUCTIVE PCA
    test_data = ift6266h12.load_npy(save_test_path)
    trans_pca = PCA(n_components = conf['n_components_trans_pca'])
    final_valid = trans_pca.fit_transform(valid_data)
    final_test = trans_pca.fit_transform(test_data)
                                        
    save_valid_path = work_dir + "valid_pca" + str(conf['nvis']) + "_dae" + str(conf['nhid']) + "_tpca" + str(conf['n_components_trans_pca']) +".npy"
    save_test_path = work_dir + "test_pca" + str(conf['nvis']) + "_dae" + str(conf['nhid']) + "_tpca" + str(conf['n_components_trans_pca']) +".npy"
    
    np.save(save_valid_path, final_valid)
    np.save(save_test_path, final_test)
    
                                        
    ###############   COMPUTE THE NEW ALC SCORE ON VALIDATION SET
    alc_2 = score(final_valid, label_data)
    
    
    ###############   OUTPUT AND RETURN THE RESULTS
    timeSpent = ((time.clock() - start) / 60.)
    print 'FINAL RESULTS (PCA-' + str(conf['nvis']) + ' DAE-' + str(conf['nhid']) + ' TransPCA-' + str(conf['n_components_trans_pca']) + ') ALC after DAE: ', alc_1, ' FINAL ALC: ', alc_2, \
            ' Computed in %5.2f min' % (timeSpent)

    return timeSpent, alc_1, alc_2
    
    


if __name__ == '__main__':
    
    main_train()