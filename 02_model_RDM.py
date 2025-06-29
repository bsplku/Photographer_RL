import pickle
import pandas as pd
import numpy as np
from rsatoolbox.model import ModelFixed
from sklearn.metrics import pairwise_distances
import os

# Set your directory
ROOT_DIRECTORY = '.'
working_directory = os.path.join(ROOT_DIRECTORY,'A_do_RSA')

# Load model embedding
data = pd.read_pickle(os.path.join(working_directory,'DQN_embeddings.pickle'))
layer_embedding = data['layer_embedding']

def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM
    Args:
        RDM 2Darray: squareform RDM
    Returns:
        1D array: upper triangular vector of the RDM
    """
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


# Create model RDMs
n_layers = 4
for i_layer in range(n_layers):
    model_RDM = pairwise_distances(layer_embedding[i_layer], metric='correlation')
    model_RDM_obj = ModelFixed('RDM', upper_tri(model_RDM))

    with open(os.path.join(working_directory,'model_RDM_layer%02d.pickle'%(i_layer+1)), 'wb') as f:
        pickle.dump({'model_RDM_obj':model_RDM_obj}, f, pickle.HIGHEST_PROTOCOL)



