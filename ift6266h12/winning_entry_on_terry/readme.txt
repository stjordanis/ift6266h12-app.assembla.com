by Li Yao

winning entry for terry.

pipeline works as following:

1. pick 20963 features from total 47236
2. rectified denoised autoencoder with 20963 inputs and 5000 outputs
3. transductive pca on test set, selecting a very small principle components

Note:
The script uses a theano sparse op that is currently not supported in the theano release.
In order to make it work, you have to use this version of theano from Yann.
hg clone https://bitbucket.org/ynd/theano. This op does not support GPU for now. So the
exp can only be done with cpu, which takes several hours with the optimal hyper-params
setup.

To run the job on the best hyper-params, just execute "./run_cpu.sh"

The step 3 is done saparately with the script "postprocess.py" which does a pca. 

The experiment will generate about 6400M data in your disk! 

Todo:
Fit this into pylearn2 framework. 
