__maintainer__ = "Li Yao"

import theano
from theano import tensor
import datetime
import copy

class LayerTrainer(object):
    """
    Only take in charge of training a specific layer. Owned by DNNTrainer.
    """
    def __init__(self, model, training_algorithm, callbacks):   

        # this is just a symbolic theano fn that transforms the actual
        # dataset when needed. this avoids dataset duplication and saves
        # lots of space.
        self.dataset_fn = None
        
        self.model = model
        self.train_algo = training_algorithm
        self.callbacks = callbacks
       
        self.epochs = 0
                 
    def train(self, dataset):
        # i know this is ugly
        data = copy.deepcopy(dataset)
        data.set_design_matrix(self.dataset_fn(dataset.get_design_matrix()))
        self.train_algo.monitoring_dataset = data
        
        if self.train_algo is None:
            # if we don not want to use SGD in pylearn2, put our algorithm here
            while self.model.train(dataset=data):
                self.epochs += 1
        else:
            # use train_algo
            self.train_algo.setup(model=self.model, dataset=data)
            epoch_start = datetime.datetime.now()

            while self.train_algo.train(dataset=data):
                epoch_end = datetime.datetime.now()
                #print 'Time of this epoch:', str(epoch_end - epoch_start)
                #import pdb;pdb.set_trace()
                monitor = self.train_algo.monitor
                #print 'Examples seen: ', monitor.examples_seen 

                epoch_start = datetime.datetime.now()

                if self.callbacks is not None:
                    for callbacks in self.callbacks:
                        callback(self.model, data, self.algorithm)

                self.epochs += 1
               
class DeepTrainer(object):
    """
    This is the master that controls all its layer trainers
    """
    def __init__(self, dataset, layer_trainers):
        """
        dataset:
        layer_trainers: list of LayerTrainer instances
        """
        self.dataset = dataset
        self.layer_trainers = layer_trainers
        self.fns = None
        self._set_symbolic_dataset_for_each_layer()
        
    def _set_symbolic_dataset_for_each_layer(self):
        """
        this maps symbolic inputs-outputs for each layer
        """
        # set inputs for each layer_trainer
        # assume that data is formatted as matrix
        X = tensor.matrix()
        
        # fns = [[l0_fn1, l0_fn2, l0_expr],[l1_fn2, l1_fn2, l1_expr],..]
        self.fns = []
        
        for ind, layer_trainer in enumerate(self.layer_trainers):
                        
            if ind == 0:
                # the bottom layer
                inputs_fn = theano.function([X], X)
                outputs_expression = layer_trainer.model.encode(X)
                outputs_fn = theano.function([X], outputs_expression)
                
            else:
                # layers above the bottom layer

                # inputs of this layer is the output of the previous layer
                inputs_fn = self.fns[ind-1][1]

                # output expr of this layer is based
                # on the expr of the previous layer
                outputs_expression = layer_trainer.model.encode(self.fns[ind-1][2])

                outputs_fn = theano.function([X], outputs_expression)
            
            entry = [inputs_fn, outputs_fn, outputs_expression]
            self.fns.append(entry)
            
            # set dataset fn to each layer
            # the idea is that we would like to avoid dataset duplication among
            # all layers. so each layer trainer only save symbolic fn in its
            # dataset. it is used to transform the actural dataset when needed.
            layer_trainer.dataset_fn = inputs_fn

    def train_supervised(self):
        raise NotImplementedError("supervised training not implemented!")
    
    def train_unsupervised(self):
        for i, layer_trainer in enumerate(self.layer_trainers):
            print "------- training layer %d ---------" % i
            layer_trainer.train(self.dataset)
           
            

        
