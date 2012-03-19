__maintainer__ = "Li Yao"

from pylearn2.autoencoder import Autoencoder, DenoisingAutoencoder
from pylearn2.models.rbm import RBM
from pylearn2.corruption import BinomialCorruptor
from pylearn2.training_algorithms.sgd import UnsupervisedExhaustiveSGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms.sgd import EpochCounter
from deep_trainer import LayerTrainer, DeepTrainer
from toy import ToyDataset
import numpy

def get_dataset():
    return ToyDataset()

def get_autoencoder(structure):
    n_input = structure[0]
    n_output = structure[1]
    conf = {
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True, 
        'act_enc': 'tanh',
        'act_dec': 'sigmoid',
        'irange': 0.001, 
    }
    return Autoencoder(**conf)

def get_layer_trainer(layer):
    # configs on sgd
    config = {'learning_rate': 0.1, 
              'cost' : MeanSquaredReconstructionError(), 
              'batch_size': 10,
              'monitoring_batches': 10,
              'monitoring_dataset': ToyDataset(), 
              'termination_criterion': EpochCounter(max_epochs=100),
              'update_callbacks': None
              }
    
    train_algo = UnsupervisedExhaustiveSGD(**config)
    model = layer
    callbacks = None
    return LayerTrainer(model, train_algo, callbacks)
    
def main():
    # get dataset
    data = get_dataset()
    design_matrix = data.get_design_matrix()
    n_input = design_matrix.shape[1]
    
    
    # build layers
    structure = [[n_input, 20],[20, 50],[50, 100]]
    layers = []
    layers.append(get_autoencoder(structure[0]))
    layers.append(get_autoencoder(structure[1]))
    layers.append(get_autoencoder(structure[2]))
    
    # construct layer trainers
    
    layer_trainers = []
    layer_trainers.append(get_layer_trainer(layers[0]))
    layer_trainers.append(get_layer_trainer(layers[1]))
    layer_trainers.append(get_layer_trainer(layers[2]))

    # init trainer that performs
    master_trainer = DeepTrainer(data, layer_trainers)
    master_trainer.train_unsupervised()


if __name__ == '__main__':
    main()
