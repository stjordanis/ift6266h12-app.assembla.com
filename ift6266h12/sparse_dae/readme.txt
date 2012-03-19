# This package includes the sparse denoising autoencoder that was used in the winning entry on Terry and Harry.
Everything is working correctly with pylearn2, including loading dataset, training models and creating the final submission.

Start by running:
python train_sparse_da.py -d toy_sparse
which takes about 3s per epoch

To make it run on Terry:
python train_sparse_da.py -d terry
which takes about 40m per epoch

Set MAX_EPOCH properly in train_sparse!

After training, use create_entry.py to do PCA and create a submission.

I may have missed something in the code. If you spot something suspicious, please let me know.

For the details of the sparse denoising autoencoder, see this paper:
Y. Dauphin, X. Glorot, Y. Bengio. Large-Scale Learning of Embeddings with Reconstruction Sampling. 
In Proceedings of the 28th International Conference on Machine Learning (ICML 2011)
