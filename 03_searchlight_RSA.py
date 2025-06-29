import pickle
import scipy.io as sio
import numpy as np
import nibabel as nib
from sklearn.metrics import pairwise_distances
import os
from rsatoolbox.util.searchlight import evaluate_models_searchlight
from rsatoolbox.inference import eval_fixed


# Set your directory
ROOT_DIRECTORY = '.'
working_directory = os.path.join(ROOT_DIRECTORY,'A_do_RSA')

# Load mask
mask3d_file = os.path.join(ROOT_DIRECTORY,'brain_mask.mat')
mask3d = sio.loadmat(mask3d_file)['mask3d']
x, y, z = mask3d.shape

# Load neural RDMs
with open(os.path.join(working_directory,'neural_RDM.pickle'), 'rb') as f:
    all_eval_score_sbj_file = pickle.load(f)
SL_RDM = all_eval_score_sbj_file['neural_RDM']

# Load model RDMs
n_layers = 4
model_RDM_obj = [None]*(n_layers)
for i_layer in range(n_layers):
    with open(os.path.join(working_directory,'model_RDM_layer%02d.pickle'%(i_layer+1)), 'rb') as f:
        model_RDM_obj[i_layer] = pickle.load(f)
    model_RDM_obj[i_layer] = model_RDM_obj[i_layer]['model_RDM_obj']

# Conduct RSA
all_eval_score_sbj = []
for i_layer in range(n_layers):
    # Exclude within-run components
    n_trials = len(all_eval_score_sbj_file['runs'])
    between_run_mask = pairwise_distances(np.expand_dims(all_eval_score_sbj_file['runs'],1),np.expand_dims(all_eval_score_sbj_file['runs'],1))!=0
    between_run_triu_mask = np.triu(np.ones((n_trials,n_trials),dtype=int),1)*between_run_mask
    vec_ids = np.where(np.triu(np.ones((n_trials,n_trials),dtype=int),1))
    vec_ids = list(zip(vec_ids[0],vec_ids[1]))
    masked_vec_ids = np.where(between_run_triu_mask)
    masked_vec_ids = list(zip(masked_vec_ids[0],masked_vec_ids[1]))
    masked_vec_loc = np.array([item in masked_vec_ids for item in vec_ids])
    model_RDM_obj[i_layer].rdm_obj.dissimilarities[:,masked_vec_loc==False] = None
    SL_RDM.dissimilarities[:,masked_vec_loc==False] = None

    # Searchlight RSA
    eval_results = evaluate_models_searchlight(SL_RDM, model_RDM_obj[i_layer], eval_fixed, method='spearman', n_jobs=3)

    # get the evaluation score for each voxel
    eval_score = [np.float32(e.evaluations.sum()) for e in eval_results]
    all_eval_score_sbj.append(eval_score)

# Save the RSA result
sio.savemat(os.path.join(working_directory,'S09_RSA_score.mat'), mdict={'eval_score': all_eval_score_sbj})



