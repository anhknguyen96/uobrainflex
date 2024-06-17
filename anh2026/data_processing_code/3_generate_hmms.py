# load data & test set LL across states calculations to determine n_states and fit glm-hmm for each subject

import os
import numpy as np
import glob
from uobrainflex.behavioranalysis import flex_hmm
import ssm
import ray
import pandas as pd
from pathlib import Path

def partition_data_by_session(inpt, y, session,data_sub):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    hmm_trials = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
    counter = 0
    # masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        hmm_trials.append(data_sub.loc[data_sub.session_identifier==sess].reset_index())
        # masks.append(mask[idx, :])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, hmm_trials

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


# base_folder = input("Enter the main directory path") + '\\'
# hmm_trials_paths = glob.glob(base_folder + 'hmm_trials\\*hmm_trials.npy')
# save_folder = base_folder
# analysis_folder_name = input("Enter the main directory path")
# csv_file_to_analyze = input("Enter csv file to analyze")
analysis_folder_name = "/home/anh/Documents/uobrainflex_test/anh2026"
csv_file_to_analyze = "om_all_batch1&2&3&4_rawrows.csv"
root_data_dir = Path(analysis_folder_name) / 'data'
analysis_result_name = Path(analysis_folder_name) / 'results'
data_path = root_data_dir / csv_file_to_analyze

data = pd.read_csv(data_path)
mouse_id_lst = data.mouse_id.unique()

col_inpts = ['freq_trans']
col_choices = ['lick_side_freq']

for m in range(len(mouse_id_lst)):
    print('mouse ' + str(mouse_id_lst[m]))
    subject = str(mouse_id_lst[m])

    #load previously calculated log likelihoods across # states to determine number of states for this subject
    LL= np.load(glob.glob(analysis_result_name / 'n_states' / subject+'*MLE_test*')[-1])
    # average across folds, and take the best iteration of fitting for each # of states
    LL = LL.mean(axis=-1).max(axis=0) 
    # find plateu of state x LL function to determine n_states
    num_states = plateu(LL)+1
    
    # load hmm_trials variable and get trial inputs and choices
    # hmm_trials = np.load(hmm_trials_paths[m],allow_pickle= True)
    # true_choices = flex_hmm.get_true_choices_from_hmm_trials(hmm_trials)
    # inpts = flex_hmm.get_inpts_from_hmm_trials(hmm_trials)
    data_sub = data.loc[(data.mouse_id==mouse_id_lst[m])&(data.lick_side_freq!=-2)].reset_index()
    data_sub.freq_trans = (data_sub.freq_trans - np.mean(data_sub.freq_trans)) / np.std(data_sub.freq_trans)
    session_arr = data_sub.session_identifier.to_numpy()
    inpt_arr = data_sub[col_inpts].to_numpy()
    choice_arr = data_sub[col_choices].to_numpy()
    inpts, true_choices, hmm_trials = partition_data_by_session(inpt_arr,choice_arr,session_arr,data_sub)
    
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
    probs,hmm_trials = flex_hmm.get_posterior_probs(hmm, hmm_trials)
    hmm = flex_hmm.permute_hmm(hmm, hmm_trials)
    probs,hmm_trials = flex_hmm.get_posterior_probs(hmm, hmm_trials, occ_thresh = .8)
    np.save(analysis_result_name / 'hmm_trials' /  subject + '_hmm_trials.npy',hmm_trials)
    np.save(analysis_result_name / 'hmms' / subject + '_hmm.npy',hmm)
