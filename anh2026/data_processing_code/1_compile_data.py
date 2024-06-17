from uobrainflex.behavioranalysis import flex_hmm
import numpy as np
from pathlib import Path

base_folder = input("Enter the main directory path") + '\\'
p=Path(base_folder)
mouse_paths =  list(p.joinpath('nwb').glob('BW*')) # list of folderpaths containing subject nwb files for analysis
for mouse_path in mouse_paths:
    print(mouse_path)
    nwbfilepaths = list(mouse_path.glob('*nwb')) #generate list of individual nwb files for a subject
    
    #generate intermediate .npy file with relevant data from nwb files for each session. These "hmm_trial" files will be used extensively
    inpts, true_choices, hmm_trials = flex_hmm.compile_choice_hmm_data(nwbfilepaths, get_behavior_measures = True)
    # hmm_trials has to be converted to numpy array before being saved
    hmm_trials_ = np.asarray(hmm_trials, dtype='object')
    np.save(p.joinpath(mouse_path.name+'_hmm_trials.npy'),hmm_trials_)
    