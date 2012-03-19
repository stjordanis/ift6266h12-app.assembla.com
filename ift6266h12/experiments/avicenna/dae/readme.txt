About the results:
- All the results are for the Avicenna dataset.
- They all start with a PCA-whiten of 75 components. Then a DAE is applied and finally a Transductive PCA layer.
- nhid is the number of hidden units for the DAE layer. Only binomial noise is explored for that test subset (with corruption_noise given in the corr column)
- ALC1 is the ALC score after the DAE and before the TransPCA
- ALC2 is the final ALC score

Here are the hyper-parameters that were constant for those test:
- act_enc : sigmoid
- act_dec : linear
- reconstruction : MSE
- batch_size : 20

