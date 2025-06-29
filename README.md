# Repository for "Neural Representations of Human Reinforcement Learning in Real-World Environments Revealed by a Deep RL Agent"
![fig](https://github.com/JuhyeonHailey/Photographer_DQN_RSA/blob/main/PhotographerRL.png?raw=true)

# Prepare for Conda environment

```bash
# Create Conda environment
conda env create -n rsa -f environment.yml

# Activate Conda environment
conda activate rsa 

# If you get an error when importing cortex, try below
python -m pip install --upgrade --no-cache-dir --force-reinstall wget
```
# Codes
You can follow the codes below step by step.

## 01_neural_RDM.py
Create neural representational dissimilarity matrices (RDMs) from fMRI data.

## 02_model_RDM.py
Create model RDMs from DQN embeddings

## 03_searchlight_RSA.py
Conduct representational similarity analysis (RSA) between the neural RDMs and model RDMs. 

## 04_RSA_group_result.py
Get the group result of RSA.

## 05_show_RSA_results.py
Get layer assignment of RSA maps and visualize. 

# Directories
## A_do_RSA
For 01_neural_RDM.py, 02_model_RDM.py, 03_searchlight_RSA.py.
Please download [beta_maps.npz](http://bspl.korea.ac.kr/Research_data/PhotographerRL/beta_maps.npz) and put in this directory.

## B_group_RSA
For 04_RSA_group_result.py.

## C_show_RSA_results
For 05_show_RSA_results.py.
Please download [RSA_score_sbjs_rndperm_l01.mat](http://bspl.korea.ac.kr/Research_data/PhotographerRL/RSA_score_sbjs_rndperm_l01.mat), [RSA_score_sbjs_rndperm_l02.mat](http://bspl.korea.ac.kr/Research_data/PhotographerRL/RSA_score_sbjs_rndperm_l02.mat), [RSA_score_sbjs_rndperm_l03.mat](http://bspl.korea.ac.kr/Research_data/PhotographerRL/RSA_score_sbjs_rndperm_l03.mat), [RSA_score_sbjs_rndperm_l04.mat](http://bspl.korea.ac.kr/Research_data/PhotographerRL/RSA_score_sbjs_rndperm_l04.mat) here and put in this directory.

### Author
>Juhyeon Lee, Ph.D. \
>jh0104lee@gmail.com 
