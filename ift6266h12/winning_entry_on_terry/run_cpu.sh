#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 \
jobman cmdline terry_da.SamplingsparseSDAEexp \
act='rect' \
batchsize=10 \
con='white' \
epochs=\[5\,10\,15\,20\,25\,30\,60\] \
featsub='/data/lisa/data/UTLC/sparse/terry_testvalid_activefeat.npy' \
lr=0.03 \
N=1 \
nepochs=60 \
new=1 \
n_hid=5000 \
ninp=20963 \
ninputs=20963 \
ones=0. \
pattern='inpnoise' \
ratio=0.005 \
regcoef=0.0001 \
savespec=1 \
scaling=0 \
seed=1 \
trans='valid' \
zeros=0.5 \
cost='MSE'
