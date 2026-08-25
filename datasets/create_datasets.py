# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import pickle
import pandas as pd

from loaddata.session import Session
from loaddata.session_info import filter_sessions
from utils.params import params
import utils.fct_data as dat
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 

#%% Dataset A (Grating sessions)

######     #    #######    #     #####  ####### #######       #    
#     #   # #      #      # #   #     # #          #         # #   
#     #  #   #     #     #   #  #       #          #        #   #  
#     # #     #    #    #     #  #####  #####      #       #     # 
#     # #######    #    #######       # #          #       ####### 
#     # #     #    #    #     # #     # #          #       #     # 
######  #     #    #    #     #  #####  #######    #       #     # 


#%% Load an example session: 
session_list            = np.array(['LPE12223_2024_06_10']) #GR
sessions,nSessions      = filter_sessions(protocols = 'GR',only_session_id=session_list,filter_noiselevel=True)
sessions[0].load_respmat()
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')

savefilename = os.path.join(os.getcwd(),'datasets','dataset_A_1_exampleses')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)

#%% Load all GR sessions, neural V1 only: 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_areas='V1',filter_noiselevel=True)
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat()
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')

savefilename = os.path.join(os.getcwd(),'datasets','dataset_A_2_V1only')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)

#%% Load all GR sessions, all areas only: 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=True)
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat()
sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')

savefilename = os.path.join(os.getcwd(),'datasets','dataset_A_3_all')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)





#%% Dataset B (Primate V1, V2 recordings, Zandvakilii, Kohn et al.)

######     #    #######    #     #####  ####### #######    ######  
#     #   # #      #      # #   #     # #          #       #     # 
#     #  #   #     #     #   #  #       #          #       #     # 
#     # #     #    #    #     #  #####  #####      #       ######  
#     # #######    #    #######       # #          #       #     # 
#     # #     #    #    #     # #     # #          #       #     # 
######  #     #    #    #     #  #####  #######    #       ######  

#%% Open data for an example session
ii_session = 2

path = 'E:\\Python\\AminData\\'
# data = scipy.io.loadmat(path+'MatlabData/mat_neural_data/'+dat.session_names[ii_session]+'.mat')['neuralData'][0][0]

# Import the data:
# spikesV1_array, spikesV2_array, stimID, trialID = dat.GetData(ii_session, path=path)
spikesV1_array, spikesV2_array, stimID, trialID = dat.GetData(ii_session, path=path,period='Stim')

#Show two example neurons: 
# import matplotlib.pyplot as plt
# fig,axes = plt.subplots(1,2,figsize=(10,5))
# axes[0].imshow(spikesV1_array[0,:,:],vmin=0,vmax=.1)
# axes[1].imshow(spikesV2_array[3,:,:],vmin=0,vmax=.1)

#Make a session object to directly relate to mouse data: 
ses         = Session(protocol='GR', animal_id=dat.session_names[ii_session], sessiondate=dat.session_names[ii_session])
ses.sessiondata  = pd.DataFrame({'protocol': ['GR'], 'sessiondate': dat.session_names[ii_session],
                                 'species': 'Macaca fascicularis', 'experimenter': 'Amin Zandvakili',
                                 'lab': 'Adam Kohn'})
sessions[0].sessiondata
Nstimuli    = 8
oris        = np.arange(0,180,180/Nstimuli)
NV1         = np.shape(spikesV1_array)[0]
NV2         = np.shape(spikesV2_array)[0]
Nrepet      = int(spikesV1_array.shape[1]/Nstimuli)
Ntrials     = spikesV1_array.shape[1]

ses.celldata  = pd.DataFrame({'roi_name': np.concatenate((np.tile(['V1'],NV1),np.tile(['V2'],NV2))),
                              'tuning_var': np.zeros(NV1+NV2)
                              })

ses.trialdata = pd.DataFrame({'Orientation': oris[stimID[::2]-1]})

idx_time = np.arange(0,spikesV1_array.shape[2],1)
idx_time = (idx_time>100) & (idx_time<1000)

V1resp                  = np.mean(spikesV1_array[:,:,idx_time], axis=2)
V2resp                  = np.mean(spikesV2_array[:,:,idx_time], axis=2)
ses.respmat             = np.concatenate((V1resp,V2resp),axis=0)

ses.respmat_videome     = np.ones(Ntrials)
ses.respmat_runspeed    = np.ones(Ntrials)
sessions = []
sessions.append(ses)
sessions = compute_pop_coupling(sessions,version='area')
sessions = compute_tuning_wrapper(sessions)

savefilename = os.path.join(os.getcwd(),'datasets','dataset_B_1_exampleses')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)

#%% Load data for all sessions
sessions = []

path = 'E:\\Python\\AminData\\'
for ii_session in range(len(dat.session_names)):
    # data = scipy.io.loadmat(path+'MatlabData/mat_neural_data/'+dat.session_names[ii_session]+'.mat')['neuralData'][0][0]

    # Import the data:
    spikesV1_array, spikesV2_array, stimID, trialID = dat.GetData(ii_session, path=path,period='Stim')

    #Make a session object: 
    ses         = Session(protocol='GR', animal_id=dat.session_names[ii_session], sessiondate=dat.session_names[ii_session])
    ses.sessiondata  = pd.DataFrame({'protocol': ['GR'], 'sessiondate': dat.session_names[ii_session],
                                 'species': 'Macaca fascicularis', 'experimenter': 'Amin Zandvakili',
                                 'lab': 'Adam Kohn'})

    Nstimuli    = 8
    oris        = np.arange(0,180,180/Nstimuli)
    NV1         = np.shape(spikesV1_array)[0]
    NV2         = np.shape(spikesV2_array)[0]
    Nrepet      = int(spikesV1_array.shape[1]/Nstimuli)
    Ntrials     = spikesV1_array.shape[1]

    ses.celldata  = pd.DataFrame({'roi_name': np.concatenate((np.tile(['V1'],NV1),np.tile(['V2'],NV2))),
                                'tuning_var': np.zeros(NV1+NV2)
                                })

    ses.trialdata = pd.DataFrame({'Orientation': oris[stimID[::2]-1]})

    idx_time = np.arange(0,spikesV1_array.shape[2],1)
    idx_time = (idx_time>100) & (idx_time<1000)

    V1resp                  = np.mean(spikesV1_array[:,:,idx_time], axis=2)
    V2resp                  = np.mean(spikesV2_array[:,:,idx_time], axis=2)
    ses.respmat             = np.concatenate((V1resp,V2resp),axis=0)

    ses.respmat_videome     = np.ones(Ntrials)
    ses.respmat_runspeed    = np.ones(Ntrials)
    sessions.append(ses)
    sessions = compute_pop_coupling(sessions,version='area')
    sessions = compute_tuning_wrapper(sessions)
    del spikesV1_array, spikesV2_array
    
savefilename = os.path.join(os.getcwd(),'datasets','dataset_B_2_all')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)


#%% Dataset C (Natural images)

######     #    #######    #     #####  ####### #######     #####  
#     #   # #      #      # #   #     # #          #       #     # 
#     #  #   #     #     #   #  #       #          #       #       
#     # #     #    #    #     #  #####  #####      #       #       
#     # #######    #    #######       # #          #       #       
#     # #     #    #    #     # #     # #          #       #     # 
######  #     #    #    #     #  #####  #######    #        #####   

#%% Load an example session: 
session_list            = np.array(['LPE13959_2025_02_24'])
sessions,nSessions      = filter_sessions(protocols = 'IM',only_session_id=session_list,filter_noiselevel=True)
sessions[0].load_respmat()
sessions = compute_pop_coupling(sessions,version='radius_500')
sessions = compute_tuning_wrapper(sessions)

savefilename = os.path.join(os.getcwd(),'datasets','dataset_C_1_exampleses')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)

#%% Load all IM sessions, neural V1 only: 
sessions,nSessions   = filter_sessions(protocols = 'IM',filter_areas='V1',filter_noiselevel=True)
for ises in range(nSessions):
    sessions[ises].load_respmat()
sessions = compute_pop_coupling(sessions,version='radius_500')
sessions = compute_tuning_wrapper(sessions)

savefilename = os.path.join(os.getcwd(),'datasets','dataset_C_2_allV1neural')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)

#%% Load all IM sessions, repeats only
sessions,nSessions   = filter_sessions(protocols = 'IM',filter_areas='V1',filter_noiselevel=True,im_ses_with_repeats=True)
for ises in range(nSessions):
    sessions[ises].load_respmat()
sessions = compute_pop_coupling(sessions,version='radius_500')
sessions = compute_tuning_wrapper(sessions)

savefilename = os.path.join(os.getcwd(),'datasets','dataset_C_3_repeatsV1neural')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)
