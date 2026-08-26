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
import scipy.io

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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
ses.session_id  = dat.session_names[ii_session]
Nstimuli    = 8
oris        = np.arange(0,180,180/Nstimuli)
NV1         = np.shape(spikesV1_array)[0]
NV2         = np.shape(spikesV2_array)[0]
Nrepet      = int(spikesV1_array.shape[1]/Nstimuli)
Ntrials     = spikesV1_array.shape[1]

ses.celldata  = pd.DataFrame({'roi_name': np.concatenate((np.tile(['V1'],NV1),np.tile(['V2'],NV2))),
                            'session_id': np.repeat(dat.session_names[ii_session],NV1+NV2)
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
    ses.session_id  = dat.session_names[ii_session]

    Nstimuli    = 8
    oris        = np.arange(0,180,180/Nstimuli)
    NV1         = np.shape(spikesV1_array)[0]
    NV2         = np.shape(spikesV2_array)[0]
    Nrepet      = int(spikesV1_array.shape[1]/Nstimuli)
    Ntrials     = spikesV1_array.shape[1]

    ses.celldata  = pd.DataFrame({'roi_name': np.concatenate((np.tile(['V1'],NV1),np.tile(['V2'],NV2))),
                                'session_id': np.repeat(dat.session_names[ii_session],NV1+NV2)
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
    del spikesV1_array, spikesV2_array
sessions = compute_pop_coupling(sessions,version='area')
sessions = compute_tuning_wrapper(sessions)

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


#%% Dataset D (Primate V1, Smith & Kohn, CRCNS pvc-11 gratings)

######     #    #######    #     #####  ####### #######     ######
#     #   # #      #      # #   #     # #          #       #     #
#     #  #   #     #     #   #  #       #          #       #     #
#     # #     #    #    #     #  #####  #####      #       #     #
#     # #######    #    #######       # #          #       #     #
#     # #     #    #    #     # #     # #          #       #     #
######  #     #    #    #     #  #####  #######    #       ######

#%% Load an example session: monkey 1, V1, drifting gratings (Smith & Kohn, CRCNS pvc-11):
path            = 'E:\\Python\\SmithKohn\\data_and_scripts\\spikes_gratings\\'
monkey          = 'monkey1'
matdata         = scipy.io.loadmat(path + 'data_%s_gratings.mat' % monkey)
EVENTS          = matdata['data'][0,0]['EVENTS']  #neurons x stimuli x trials, spike times (s) rel. to stim onset

NNeurons,Nstim,Ntrialsperstim    = EVENTS.shape
oris            = np.arange(0,360,360/Nstim)    #12 directions, spaced 30 deg apart

t_resp_start    = 0    #response window start (s), relative to stimulus onset
t_resp_stop     = 1    #response window stop (s), relative to stimulus onset

Ntrials         = Nstim * Ntrialsperstim
respmat         = np.full((NNeurons,Ntrials),np.nan)
trialoris       = np.repeat(oris,Ntrialsperstim)

#Convert spike times to average firing rate (Hz) in the 0-1s poststimulus window, for each neuron and trial:
for iN in range(NNeurons):
    for istim in range(Nstim):
        for irep in range(Ntrialsperstim):
            spiketimes                 = EVENTS[iN,istim,irep].flatten()
            itrial                     = istim * Ntrialsperstim + irep
            respmat[iN,itrial]         = np.sum((spiketimes>=t_resp_start) & (spiketimes<t_resp_stop)) / (t_resp_stop-t_resp_start)

#Make a session object to directly relate to mouse data:
ses             = Session(protocol='GR', animal_id=monkey, sessiondate=monkey)
ses.sessiondata = pd.DataFrame({'protocol': ['GR'], 'sessiondate': monkey,
                                 'species': 'Macaca fascicularis', 'experimenter': 'Matthew Smith',
                                 'lab': 'Adam Kohn'})
ses.session_id  = monkey

ses.celldata    = pd.DataFrame({'roi_name': np.tile(['V1'],NNeurons),
                                'session_id': np.repeat(monkey,NNeurons)
                                 })

ses.trialdata   = pd.DataFrame({'Orientation': trialoris})

ses.respmat             = respmat
ses.respmat_videome     = np.ones(Ntrials)
ses.respmat_runspeed    = np.ones(Ntrials)

sessions = [ses]
# sessions = []
# sessions.append(ses)
sessions = compute_pop_coupling(sessions,version='area')
sessions = compute_tuning_wrapper(sessions)
savefilename = os.path.join(os.getcwd(),'datasets','dataset_D_1_exampleses')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)

#%% Load all datasets:
path            = 'E:\\Python\\SmithKohn\\data_and_scripts\\spikes_gratings\\'
sessions = []
for ises,monkey in enumerate(['monkey1','monkey2','monkey3']):

    matdata         = scipy.io.loadmat(path + 'data_%s_gratings.mat' % monkey)
    EVENTS          = matdata['data'][0,0]['EVENTS']  #neurons x stimuli x trials, spike times (s) rel. to stim onset

    NNeurons,Nstim,Ntrialsperstim    = EVENTS.shape
    oris            = np.arange(0,360,360/Nstim)    #12 directions, spaced 30 deg apart

    t_resp_start    = 0    #response window start (s), relative to stimulus onset
    t_resp_stop     = 1    #response window stop (s), relative to stimulus onset

    Ntrials         = Nstim * Ntrialsperstim
    respmat         = np.full((NNeurons,Ntrials),np.nan)
    trialoris       = np.repeat(oris,Ntrialsperstim)

    #Convert spike times to average firing rate (Hz) in the 0-1s poststimulus window, for each neuron and trial:
    for iN in range(NNeurons):
        for istim in range(Nstim):
            for irep in range(Ntrialsperstim):
                spiketimes                 = EVENTS[iN,istim,irep].flatten()
                itrial                     = istim * Ntrialsperstim + irep
                respmat[iN,itrial]         = np.sum((spiketimes>=t_resp_start) & (spiketimes<t_resp_stop)) / (t_resp_stop-t_resp_start)

    #Make a session object to directly relate to mouse data:
    ses             = Session(protocol='GR', animal_id=monkey, sessiondate=monkey)
    ses.sessiondata = pd.DataFrame({'protocol': ['GR'], 'sessiondate': monkey,
                                    'species': 'Macaca fascicularis', 'experimenter': 'Matthew Smith',
                                    'lab': 'Adam Kohn'})
    ses.session_id  = monkey
    ses.celldata    = pd.DataFrame({'roi_name': np.tile(['V1'],NNeurons),
                                    'session_id': np.repeat(monkey,NNeurons)
                                    })

    ses.trialdata   = pd.DataFrame({'Orientation': trialoris})

    ses.respmat             = respmat
    ses.respmat_videome     = np.ones(Ntrials)
    ses.respmat_runspeed    = np.ones(Ntrials)

    sessions.append(ses)
    del matdata, EVENTS

sessions = compute_pop_coupling(sessions,version='area')
sessions = compute_tuning_wrapper(sessions)
savefilename = os.path.join(os.getcwd(),'datasets','dataset_D_2_all')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)


#%% Dataset E (Allen Institute Visual Coding Neuropixels, Functional Connectivity, drifting gratings 75 repeats)

######     #    #######    #     #####  ####### #######     #######
#     #   # #      #      # #   #     # #          #       #
#     #  #   #     #     #   #  #       #          #       #
#     # #     #    #    #     #  #####  #####      #       #####
#     # #######    #    #######       # #          #       #
#     # #     #    #    #     # #     # #          #       #
######  #     #    #    #     #  #####  #######    #       #######

#%% Load an example session: mouse V1+HVAs, drifting gratings (Allen Institute Visual Coding Neuropixels,
#session_type == 'functional_connectivity', which shows each grating orientation/contrast combination for
#75 repeats, unlike the sparser 'brain_observatory_1.1' session type (~15 repeats).
#Session 767871931 was picked because it has good unit yield in V1 (VISp) and PM (VISpm), plus AL/AM/RL/LI.
#allensdk pins pandas==1.5.3/numpy<1.24, which would downgrade the rest of this env, so it was installed
#with --no-deps (keeping the existing pandas/numpy) plus its other deps installed separately, pinning
#psycopg2-binary==2.9.9, SimpleITK==2.3.1 and aiohttp==3.9.5 (only versions with prebuilt py3.8 wheels).
#SimpleITK/argschema/scikit-build/glymur are only used by allensdk's atlas/registration tooling, not the
#ecephys spike-data path used here, but SimpleITK is still imported at module load time so is required.
cache_dir       = 'E:\\Python\\AllenNeuropixels\\'
manifest_path   = os.path.join(cache_dir,'manifest.json')
cache           = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_id      = 767871931
allensession    = cache.get_session_data(session_id)
session_id      = 'AllenNP_%d' % session_id
visualareas     = ['VISp','VISpm','VISal','VISam','VISrl','VISli']
areamap         = {'VISp':'V1','VISpm':'PM','VISal':'AL','VISam':'AM','VISrl':'RL','VISli':'LI'}

units           = allensession.units[allensession.units['ecephys_structure_acronym'].isin(visualareas)]
unit_ids        = units.index.to_numpy()
NNeurons        = len(unit_ids)

stimdata        = allensession.stimulus_presentations
stimdata        = stimdata[stimdata['stimulus_name']=='drifting_gratings_75_repeats']
Ntrials         = len(stimdata)
starts          = stimdata['start_time'].to_numpy()

t_resp_start,t_resp_stop = 0,1     #response window (s), relative to stimulus onset

#Convert spike times to average firing rate (Hz) in the 0-1s poststimulus window, for each neuron and trial:
respmat         = np.full((NNeurons,Ntrials),np.nan)
for iN,uid in enumerate(unit_ids):
    spiketimes          = allensession.spike_times[uid]
    idx_start           = np.searchsorted(spiketimes,starts+t_resp_start,side='left')
    idx_stop            = np.searchsorted(spiketimes,starts+t_resp_stop,side='left')
    respmat[iN,:]       = (idx_stop-idx_start) / (t_resp_stop-t_resp_start)

#Average running speed (cm/s) in the same poststimulus window, for each trial:
rundata             = allensession.running_speed
runtimes            = (rundata['start_time'].to_numpy() + rundata['end_time'].to_numpy()) / 2
runspeed            = rundata['velocity'].to_numpy()
order               = np.argsort(runtimes)
runtimes,runspeed   = runtimes[order],runspeed[order]

respmat_runspeed    = np.full(Ntrials,np.nan)
for it,t0 in enumerate(starts):
    idx_start                  = np.searchsorted(runtimes,t0+t_resp_start,side='left')
    idx_stop                   = np.searchsorted(runtimes,t0+t_resp_stop,side='left')
    respmat_runspeed[it]       = np.mean(runspeed[idx_start:idx_stop])

#Make a session object to directly relate to mouse data:
ses             = Session(protocol='GR', animal_id=session_id, sessiondate=session_id)
ses.sessiondata = pd.DataFrame({'protocol': ['GR'], 'sessiondate': session_id,
                                 'species': 'Mus musculus', 'experimenter': 'Allen Institute',
                                 'lab': 'Allen Institute (Visual Coding Neuropixels, Functional Connectivity)'})
ses.session_id = session_id
ses.celldata    = pd.DataFrame({'roi_name': units['ecephys_structure_acronym'].map(areamap).to_numpy(),
                                 'session_id': np.repeat(session_id,NNeurons)
                                 })

ses.trialdata   = pd.DataFrame({'Orientation': stimdata['orientation'].to_numpy(dtype=float),
                                 'Contrast': stimdata['contrast'].to_numpy(dtype=float)})

ses.respmat             = respmat
ses.respmat_runspeed    = respmat_runspeed
ses.respmat_videome     = np.ones(Ntrials)

sessions = [ses]
sessions = compute_pop_coupling(sessions,version='area')
sessions = compute_tuning_wrapper(sessions)

savefilename = os.path.join(os.getcwd(),'datasets','dataset_E_1_exampleses')
np.savez(savefilename + '.npz',sessions=sessions,allow_pickle=True)


#%%









