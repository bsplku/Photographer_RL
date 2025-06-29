import pickle
import scipy.io as sio
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like
from nilearn.reporting import get_clusters_table
from nilearn import image
import os
import copy
import cortex

# Set your directory
ROOT_DIRECTORY = '.'
working_directory = os.path.join(ROOT_DIRECTORY,'C_show_RSA_results')

# Load mask
mask3d_file = os.path.join(ROOT_DIRECTORY,'mni_152_gm_mask_3mm.nii')
mask3d_obj = nib.load(mask3d_file)
mask3d = np.array(mask3d_obj.dataobj)
x, y, z = mask3d.shape

# Load searchlight centers of all subjects
with open(os.path.join(ROOT_DIRECTORY,'center_sbjs.mat'), 'rb') as f:
    centers = pickle.load(f)
centers = centers['centers']

# Make 'or' mask of centers
centers_or = np.array(list(set(np.concatenate(centers))))
centers_or = centers_or[centers_or!=0]
centers_or_1d = np.zeros((x*y*z))
centers_or_1d[centers_or] = 1
centers_or_3d = centers_or_1d.reshape(x,y,z)
n_vox = np.sum(centers_or_3d,dtype=int)

# Set threshold and minimum cluster size
p_thr=0.005
clst_size = 30
n_layers = 4

# Set threhold by permutation result
t_score = [None]*(n_layers)
t_score_thr = [None]*(n_layers)
for i_layer in range(n_layers):
    # Load t-test result
    eval_score_file = sio.loadmat(os.path.join(working_directory,'RSA_score_sbjs_layer%d.mat'%(i_layer+1)))
    t_score[i_layer] = eval_score_file['t'].squeeze()
    tmpt = copy.copy(t_score[i_layer])
    # Load permutation result
    eval_rndperm_file = sio.loadmat(os.path.join(working_directory,'RSA_score_sbjs_rndperm_l%02d.mat'%(i_layer+1)))
    t_rndperm = eval_rndperm_file['t'].squeeze()
    # Set threshold
    ub = np.percentile(t_rndperm,100-p_thr*100,axis=0)
    tmpt = tmpt*(tmpt >= ub)
    tmpt[np.isnan(tmpt)] = 0
    t_score_thr[i_layer] = tmpt
    # Save the thresholded result
    t_score_thr_3d = np.zeros((x,y,z))
    t_score_thr_3d[np.where(centers_or_3d)] = t_score_thr[i_layer]
    plot_img = new_img_like(nib.load(mask3d_file), t_score_thr_3d)
    nib.save(plot_img, os.path.join(working_directory,'RSA_score_sbjs_tscore_l%02d.nii'%(i_layer+1)))

# Clusterize
t_score_thr_layers = np.zeros((n_layers, n_vox))
for i_layer in range(n_layers):
    # Get clusters
    clst_table, clst_maps_img = get_clusters_table(os.path.join(working_directory,'RSA_score_sbjs_tscore_l%02d.nii'%(i_layer+1)), 0, cluster_threshold=clst_size, return_label_maps=True, two_sided=False)

    # Select only positive clusters
    if len(clst_table)==0:
        t_score_thr_layers[i_layer,:] = t_score_thr[i_layer]*0.0
    elif len(np.where(clst_table['Peak Stat']>0)[0])==0:
        t_score_thr_layers[i_layer,:] = t_score_thr[i_layer]*0.0
    else:
        t_score_thr_layers[i_layer,:] = t_score_thr[i_layer] *(clst_maps_img[0].get_fdata()[centers_or_3d==1]>0)

    # Save positive clusters
    if len(clst_table)>0:
        nib.save(clst_maps_img[0], os.path.join(working_directory,'RSA_score_sbjs_tscore_l%02d_clstlabel.nii'%(i_layer+1)))

# Calculate and show the layer assignment map
# Find the winner layer for each voxel
loc_assign = np.argmax(t_score_thr_layers*(t_score_thr_layers>0), axis=0)
RSA_assign = (loc_assign+1) * (t_score_thr_layers[loc_assign, np.arange(len(loc_assign))]>0)
RSA_assign_3d = np.zeros((x,y,z))
RSA_assign_3d[np.where(centers_or_3d)] = RSA_assign

# Save the assignment result
plot_img = new_img_like(nib.load(mask3d_file), RSA_assign_3d)
savepath = os.path.join(working_directory,'RSA_score_layer_assignment.nii')
nib.save(plot_img, savepath)

# Resample the assignment result to map to a surface
affine = np.array([[   3. ,   -0. ,   -0. ,  -94.5],
                       [  -0. ,    3. ,   -0. , -130.5],
                       [   0. ,    0. ,    3. ,  -76.5],
                       [   0. ,    0. ,    0. ,    1. ]])

surface_map_new = image.resample_to_img(savepath, os.path.join(working_directory,'reference.nii.gz'), interpolation="nearest").get_fdata()
surface_map_new[(surface_map_new==0)] = np.nan
surface_map_new = nib.Nifti1Image(surface_map_new,affine)
savepath_new = savepath.split('.nii')[0]+'_rsmp.nii'
nib.save(surface_map_new,savepath_new)

# Prepare for visualization
vols={}
vols['Assgn'] = cortex.Volume(savepath_new, '2009c', 'fullhead_v64', vmin=1, vmax=n_layers,cmap='Purples')

# Plot interactive viewer
cortex.webgl.show(vols, overlays_visible=('rois', 'sulci'), recache=True, autoclose=True, overlay_file=os.path.join(working_directory,'overlays_mix3.svg'))



