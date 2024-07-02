# load data & test set LL across states calculations to determine n_states and fit glm-hmm for each subject

import os
import numpy as np
import glob
from uobrainflex.behavioranalysis import flex_hmm
import ssm
import ray
import pandas as pd
from pathlib import Path
import scipy

def get_inpts_and_choices(hmm_trials,col_inpts,col_choices):
    inpts=list([])
    true_choices = list([])
    for session in hmm_trials:
        stim = session[col_inpts].values
        # add constant bias
        these_inpts = [ np.hstack((stim,np.array([np.ones(len(stim))]).T)) ]
        inpts.extend(these_inpts)

        choices = session[col_choices].values
        true_choices.extend([choices])
    return inpts, true_choices
def plateu(data,threshold=.001):
    ind = np.where(np.diff(data)>threshold)[0]
    ind[np.argmax(data[ind+1])]
    return ind[np.argmax(data[ind+1])]+1

@ray.remote
def MLE_hmm_fit(subject, num_states, training_inpts, training_choices):
    ## Fit GLM-HMM with MAP estimation:
    # Set the parameters of the GLM-HMM
    obs_dim = training_choices[0].shape[1]          # number of observed dimensions
    num_categories = len(np.unique(np.concatenate(training_choices)))    # number of categories for output
    input_dim = training_inpts[0].shape[1]                                    # input dimensions

    TOL = 10**-4
    
    N_iters = 1000

    hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                       observation_kwargs=dict(C=num_categories), transitions="standard")
    #fit on training data
    train_ll = hmm.fit(training_choices, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)   
    train_ll = train_ll[-1]/np.concatenate(training_inpts).shape[0]       
    return hmm, train_ll

analysis_folder_name = "/home/anh/Documents/uobrainflex_test/anh2026"
analysis_result_name = Path(analysis_folder_name) / 'results'
hmm_path = analysis_result_name / 'hmm_trials'
hmm_save = analysis_result_name / 'hmms'
if not os.path.exists(hmm_save):
    os.makedirs(hmm_save)


hmm_trials_paths = glob.glob(str(hmm_path) + '/' + '*hmm_trials.npy')

col_inpts = ['pfail','stim', 'stim:pfail', 'pchoice']
col_choices = ['lick_side_freq']

for m in range(len(hmm_trials_paths)):
    subject = os.path.basename(hmm_trials_paths[m])[:4]
    print('mouse ' + subject)

    #load previously calculated log likelihoods across # states to determine number of states for this subject
    LL= np.load(glob.glob(str(analysis_result_name / 'n_states')+'/'+subject+'*MLE_test*')[0])
    # average across folds, and take the best iteration of fitting for each # of states
    LL = LL.mean(axis=-1).max(axis=0)
    # find plateu of state x LL function to determine n_states
    num_states = plateu(LL)+1

    # load hmm_trials variable and get trial inputs and choices
    hmm_trials = np.load(hmm_trials_paths[m],allow_pickle= True)
    inpts, true_choices = get_inpts_and_choices(hmm_trials,col_inpts,col_choices)

    mouse_hmms=[]
    mouse_LLs=[]
    processes=[]
    # fit 10 hmms and select best
    for z in range(10):
        processes.append(MLE_hmm_fit.remote(subject, int(num_states), inpts, true_choices))

    data = np.array([ray.get(p) for p in processes])

    # select HMM with highest log likelihood
    hmm = data[np.argmax(data[:,1]),0]

    # input posterior probabilities to hmm_trials, and permute the hmm to match states across subjects
    probs,hmm_trials = flex_hmm.get_posterior_probs_om(hmm, hmm_trials,inpts,true_choices)
    hmm = flex_hmm.permute_hmm_om(hmm, hmm_trials)
    probs,hmm_trials = flex_hmm.get_posterior_probs_om(hmm, hmm_trials,inpts,true_choices, occ_thresh = .8)
    # hmm_trials has to be converted to numpy array before being saved
    hmm_trials_ = np.asarray(hmm_trials, dtype='object')
    np.save(hmm_path / (subject + '_hmm_trials.npy'),hmm_trials_)
    np.save(hmm_save / (subject + '_hmm.npy'),hmm)
