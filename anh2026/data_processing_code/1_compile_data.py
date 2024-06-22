from uobrainflex.behavioranalysis import flex_hmm
import numpy as np
from pathlib import Path
import os
import pandas as pd
import scipy
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

analysis_folder_name = "/home/anh/Documents/uobrainflex_test/anh2026"
csv_file_to_analyze = "om_all_batch1&2&3&4_rawrows.csv"

root_data_dir = Path(analysis_folder_name) / 'data'
analysis_result_name = Path(analysis_folder_name) / 'results'
data_path = root_data_dir / csv_file_to_analyze
hmm_trials_save = analysis_result_name / 'hmm_trials'
if not os.path.exists(hmm_trials_save):
    os.makedirs(hmm_trials_save)

om = pd.read_csv(data_path)
# clean data
data_batch12 = om.loc[(om.mouse_id < 13) & (om.lick_side_freq != -2) & (om.prev_choice != -2) & (om.prev_reward_prob == 0.5) & (om.prev_choice2 != -2)]
data_batch34 = om.loc[(om.mouse_id > 13) & (om.prev_om_gen == 0) & (om.lick_side_freq != -2) & (om.prev_choice != -2) & (om.prev_reward_prob == 0.5) & (om.prev_choice2 != -2)]
om_cleaned = pd.concat[(data_batch12,data_batch34)].reset_index()
# now take care of predictors
index = om_cleaned.index
om_cleaned['prev_failure'] = om_cleaned['prev_failure'].astype('int')
om_cleaned['mouse_id'] = om_cleaned['mouse_id'].astype(str)
om_cleaned['z_freq_trans'] = om_cleaned['freq_trans'].copy()
om_cleaned['z_prev_choice'] = om_cleaned['prev_choice'].copy()
for session_no in om_cleaned.session_identifier.unique():
    # get indices of trials in the session
    session_no_index = list(index[(om_cleaned['session_no'] == session_no)])
    # z score predictors on a session basis
    om_cleaned.loc[session_no_index, 'z_freq_trans'] = scipy.stats.zscore(
        om_cleaned.loc[session_no_index, 'freq_trans'])
    om_cleaned.loc[session_no_index, 'z_prev_choice'] = scipy.stats.zscore(
        om_cleaned.loc[session_no_index, 'prev_choice'])

# since min/max freq_trans is -1.5/1.5
bin_lst = np.arange(-1.55,1.6,0.1)
bin_name=np.round(np.arange(-1.5,1.6,.1),2)
# get binned freqs for psychometrics
om_cleaned["binned_freq"] = pd.cut(om_cleaned.freq_trans, bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)

col_inpts = ['z_freq_trans']
col_choices = ['lick_side_freq']
mouse_id_lst = om_cleaned.mouse_id.unique()
for m in range(len(mouse_id_lst)):
    print('mouse ' + str(mouse_id_lst[m]))
    subject = str(mouse_id_lst[m])

    data_sub = om_cleaned.loc[(om_cleaned.mouse_id == mouse_id_lst[m])].reset_index()
    session_arr = data_sub.session_identifier.to_numpy()
    inpt_arr = data_sub[col_inpts].to_numpy()
    choice_arr = data_sub[col_choices].to_numpy()
    _, _, hmm_trials = partition_data_by_session(inpt_arr, choice_arr, session_arr)

    # hmm_trials has to be converted to numpy array before being saved
    hmm_trials = np.asarray(hmm_trials, dtype='object')
    np.save(hmm_trials_save / (subject + '_hmm_trials.npy'),hmm_trials)
    