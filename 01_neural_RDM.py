import pickle
import scipy.io as sio
import numpy as np
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
import os

# Set your directory
ROOT_DIRECTORY = '.'
working_directory = os.path.join(ROOT_DIRECTORY,'A_do_RSA')

# Load mask
mask3d_file = os.path.join(working_directory,'brain_mask.mat')
mask3d = sio.loadmat(mask3d_file)['mask3d']
x, y, z = mask3d.shape

# Load neural data
data = np.load(os.path.join(working_directory,'beta_maps.npz'))

all_run = data['run']
all_x = data['X']
all_x_2d = all_x.reshape([all_x.shape[0], -1])
all_x_2d = np.nan_to_num(all_x_2d)

# Create searchlight-based neural RDMs
# In order for it to be a proper center at least 50% of the neighboring voxels needs to be within the brain mask (threshold=0.5)
threshold = 0.5
r=3

centers, neighbors = get_volume_searchlight(mask3d, radius=r, threshold=threshold)

# Save neural RDMs
SL_RDM = get_searchlight_RDMs(all_x_2d, centers, neighbors, np.arange(np.size(all_x_2d,0)), method='correlation')
with open(os.path.join(working_directory,'neural_RDM.pickle'), 'wb') as f:
    pickle.dump({'neural_RDM':SL_RDM,'centers':centers,'runs':all_run}, f, pickle.HIGHEST_PROTOCOL)


