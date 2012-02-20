
# local imports
import pylearn2
from pylearn2.base import StackedBlocks
from pylearn2.autoencoder import DenoisingAutoEncoder
from pylearn2.corruption import BinomialCorruptor





""" Train saves the whole object in pickle
your class could have two functions, one pre-train and one fine-tune
pretrain runs the train for each layer once

then
"""


class CAE():


    def __init__():

        cae = build_stacked_ae()
        self.tain_list = []
        for layer in cae.layers()
            train_obj = Train(dataset = dataset, model = layer, algorithm = algorithm)
            slef.train_list.append(train_obj)

    def __call__():

        return self.train_list


class CAE2():

    def __init__(self, corruption_levels, nvis, nhids, corruptor, act_enc, act_dec):

        self.corruption_levels = corruption_levels
        self.nlayers = len(corruption_levels)
        blocks = StackedBlocks([])

        # First CAE layer
        input = input
        layer = DenoisingAutoEncoder(corruption_level = self.corruption_levels[0],
                                        nvis = nvis
                                        nhid = nhids[0])
        blocks.append(layer)

        # Second CAE laye
        input = blocks.function(repr_index = 0)
        layer = DenoisingAutoEncoder(corruption_level = self.corruption_levels[0],
                                        nvis = nhids[0]
                                        nhid = nhids[1])
        blocks.append(layer)






    def train:

        for layer in self.blocs.layers:
            if layer
            train_ob:q
            j = Train(dataset = ds)


corruptor = BinomialCorruptor(corruption_level = 0.5)
layer1 = DenoisingAutoEncoder(nvis = 300, nhid = 400, irange = 0.05,
                                corruptor = corruptor,
                                act_enc = "tanh", act_dec = "tanh")


#TODO it would be better if StackedBlocks layers could be accessed by key
block = StackedBlocks(corruptor)



