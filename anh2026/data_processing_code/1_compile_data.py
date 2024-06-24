from uobrainflex.behavioranalysis import flex_hmm
import numpy as np
from pathlib import Path
import os
import pandas as pd
import scipy

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session
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
csv_file_to_analyze = "om_all_batch1&2&3&4_processed.csv"
processed_file_toload = 'all_animals_concat.npz'

root_data_dir = Path(analysis_folder_name) / 'data'
analysis_result_name = Path(analysis_folder_name) / 'results'
data_path = root_data_dir / csv_file_to_analyze
processed_data_path = root_data_dir / processed_file_toload
hmm_trials_save = analysis_result_name / 'hmm_trials'
if not os.path.exists(hmm_trials_save):
    os.makedirs(hmm_trials_save)

# get processed dataframe
om = pd.read_csv(data_path)
# since min/max freq_trans is -1.5/1.5
bin_lst = np.arange(-1.55,1.6,0.1)
bin_name=np.round(np.arange(-1.5,1.6,.1),2)
# get binned freqs for psychometrics
om["binned_freq"] = pd.cut(om.freq_trans, bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)

# get processed predictors
col_inpts = ['pfail','stim', 'stim:pfail', 'pchoice']
col_choices = ['lick_side_freq']
# trial info cols
col_toparse = ['binned_freq','lick_side_freq','mouse_id','session_identifier']
inpt, y, session = load_data(processed_data_path)
df_tmp = pd.DataFrame(data=inpt, columns=col_inpts)

# concatenated the trial info dataframe and the predictors dataframe
hmm_df = pd.concat([df_tmp,om[col_toparse]],axis=1)
mouse_id_lst = hmm_df.mouse_id.unique()
for m in range(len(mouse_id_lst)):
    print('mouse ' + str(mouse_id_lst[m]))
    subject = str(mouse_id_lst[m])

    data_sub = hmm_df.loc[(hmm_df.mouse_id == mouse_id_lst[m])].reset_index()
    session_arr = data_sub.session_identifier.to_numpy()
    inpt_arr = data_sub[col_inpts].to_numpy()
    choice_arr = data_sub[col_choices].to_numpy()
    _, _, hmm_trials = partition_data_by_session(inpt_arr, choice_arr, session_arr,data_sub)

    # hmm_trials has to be converted to numpy array before being saved
    hmm_trials = np.asarray(hmm_trials, dtype='object')
    np.save(hmm_trials_save / (subject + '_hmm_trials.npy'),hmm_trials)
    