import pickle
import scipy.io as sio
import numpy as np
import nibabel as nib
from nilearn.image import new_img_like
from nilearn.image import smooth_img
import scipy.stats as stats
import os

# Set your directory
ROOT_DIRECTORY = '.'
working_directory = os.path.join(ROOT_DIRECTORY,'B_group_RSA')

# Load mask
mask3d_file = os.path.join(ROOT_DIRECTORY,'mni_152_gm_mask_3mm.nii')
mask3d_obj = nib.load(mask3d_file)
mask3d = np.array(mask3d_obj.dataobj)
x, y, z = mask3d.shape

# List of subjects
id_include = [ 'S09', 'S10', 'S16', 'S17', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', \
             'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S41', 'S42', 'S45', 'S47', 'S48', 'S49', 'S50', 'S51']
n_sbjs = len(id_include)
# Spatial smoothing parameter
FWHM = 6

# Load RSA result
n_layers = 4
all_eval_score = [None] * (n_layers)
for  sbj in id_include:
    all_eval_score_sbj_file = sio.loadmat(os.path.join(working_directory,'%s_RSA_score.mat'%sbj))
    all_eval_score_sbj = all_eval_score_sbj_file['eval_score']

    for i_layer in range(n_layers):
        # Fisher's r to z
        score = np.arctanh(all_eval_score_sbj[i_layer])
        if type(all_eval_score[i_layer]) != list:
            all_eval_score[i_layer] = [score]
        else:
            all_eval_score[i_layer].append(score)

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


# Smoothing and t-test on RSA results
all_eval_score_3d = [None]*(n_layers)
all_eval_score_inmask = [None]*(n_layers)
t_eval_score = [None]*(n_layers)
p_eval_score = [None]*(n_layers)
for i_layer in range(n_layers):
    all_eval_score_3d[i_layer] = [None]*n_sbjs
    all_eval_score_inmask[i_layer] = [None]*n_sbjs
    for i_sbj in range(n_sbjs):
        score_1d = np.zeros([x * y * z])
        score_1d[centers[i_sbj]] = all_eval_score[i_layer][i_sbj]
        score_3d = score_1d.reshape(x,y,z)
        # Smoothing
        score_3d_img = new_img_like(nib.load(mask3d_file), score_3d)
        score_3d = smooth_img(score_3d_img, FWHM).get_fdata()

        all_eval_score_3d[i_layer][i_sbj] = score_3d
        all_eval_score_inmask[i_layer][i_sbj] = score_3d[np.where(centers_or_3d)]

    all_eval_score_3d[i_layer] = np.moveaxis(np.asarray(all_eval_score_3d[i_layer]),0,-1)
    all_eval_score_inmask[i_layer] = np.vstack(all_eval_score_inmask[i_layer])

    # One-sample t-test
    t_eval_score[i_layer],p_eval_score[i_layer] = stats.ttest_1samp(all_eval_score_inmask[i_layer],popmean=0,alternative='greater')

    sio.savemat(os.path.join(working_directory,'RSA_score_sbjs_layer%d.mat'%(i_layer+1)), \
                mdict={'score': all_eval_score_inmask[i_layer], 't':t_eval_score[i_layer], 'p':p_eval_score[i_layer]})



# Random permutation on signs of t-test results
n_perms = 1000

all_eval_score_inmask = [None]*(n_layers)
t_eval_score = [None]*(n_layers)
p_eval_score = [None]*(n_layers)

for i_layer in range(n_layers):
    # Load original t-test result
    eval_score_file = sio.loadmat(os.path.join(working_directory,'RSA_score_sbjs_layer%d.mat'%(i_layer+1)))
    all_eval_score_inmask[i_layer] = eval_score_file['score']

    # Random permutation (it takes long)
    t_perms = np.zeros((n_perms,np.size(all_eval_score_inmask[i_layer],1)))
    for i_v in range(np.size(t_perms,1)):
        if i_v%1000==0:
            print("layer %d voxel %d"%(i_layer+1,i_v+1))

        for i_perm in range(n_perms):
            rnd_int = np.random.randint(0,2,np.size(all_eval_score_inmask[i_layer],0))
            rnd_sign = 2*(rnd_int-0.5)
            all_eval_score_rand = rnd_sign*all_eval_score_inmask[i_layer][:,i_v]
            t_perms[i_perm,i_v], _ = stats.ttest_1samp(all_eval_score_rand,popmean=0,alternative='greater')

    # Save the permutation result
    sio.savemat(os.path.join(working_directory,'RSA_score_sbjs_rndperm_l%02d.mat'%(i_layer+1)), mdict={'t': t_perms})


