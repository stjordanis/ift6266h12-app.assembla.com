from RunPCASylvester import *
from RunCAESylvester import *

def run_pipeline():
  pca_output = "layers/npy/"
  cae_yamls = ["yamls/cae1.yaml", "yamls/cae2.yaml", "yamls/cae3.yaml", "yamls/cae4.yaml"]
  #In the first layer apply PCA and whitening
  run_standard_PCA(pca_output)
  #In the second and third layer run two stacked CAE on PCA's PC's
  run_stack(cae_yamls)
  #In the last layer run Transductive PCA
  run_transductive_PCA(pca_output)

if __name__== '__main__':
  run_pipeline()
