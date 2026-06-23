# -*- coding: utf-8 -*-
"""
This script fits receptive fields to the average response triggered image 
and saves the results for each session
Matthijs Oude Lohuis, 2023-2025, Champalimaud Center
"""
# %% # Imports
# Import general libs

# Import personal lib funcs
from loaddata.session_info import filter_sessions
from utils.rf_lib import estimate_rf_IM

# %% Load IM session with receptive field mapping ################################################
sessions, nSessions = filter_sessions(protocols='IM',only_animal_id=['LPE10885','LPE11086'])

for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

#%% Get response-triggered frame for cells and then estimate receptive field from that:
for ses in sessions:    # iterate over sessions
    rf_data    = estimate_rf_IM(ses,show_fig=False)
    outfilepath = ses.sessiondata_path.replace('sessiondata','IMrfdata')
    rf_data.to_csv(outfilepath, sep=',')


