# -*- coding: utf-8 -*-
"""
This script analyzes noise correlations in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import pickle

from utils.params import params
from utils.plot_lib import *

figdir = os.path.join(params['figdir'],'multiarea','popcoupling')

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_A_1_GRexampleses' + '.npz'),allow_pickle=True)
for key in data.keys():
    exec(key+'=data[key]')
sessions = data['sessions']

#%% How are neurons are coupled to the population rate of different areas:
areas = np.array(['V1', 'PM','AL','RSP'])
nareas = len(areas)

for ises,ses in enumerate(sessions):
    resp        = zscore(ses.respmat,axis=1)
    N           = len(ses.celldata)
    for iarea,area in enumerate(areas):
        idx_N = np.where(ses.celldata['roi_name']==area)[0]
        poprate = np.nanmean(resp[idx_N,:],axis=0)

        for iN in range(N): #correlate the activity to the population average of that area (minus itself if from that area)
            if iN in idx_N:
                ses.celldata.loc[iN,'pop_coupling_%s' % area]  = np.corrcoef(resp[iN,:],np.mean(resp[np.setdiff1d(idx_N,iN),:], axis=0))[0,1]
            else:
                ses.celldata.loc[iN,'pop_coupling_%s' % area]  = np.corrcoef(resp[iN,:],poprate)[0,1]

#%% Concatenate data:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Make a scatter plot of the population coupling to each areas mean activity
varslist = ['pop_coupling_'+area for area in areas]
# pairplot = sns.pairplot(celldata, vars=varslist,markers='.',diag_kws={'alpha': 0.5})
pairplot = sns.pairplot(celldata, vars=varslist,markers='.',plot_kws={'s': 0.5},hue='roi_name')
pairplot.fig.set_size_inches(5*cm, 5*cm)
my_savefig(fig=pairplot.fig, figdir=figdir, filename='cross_correlation_%s'  % ses.session_id)

#%%
corrmat = celldata[varslist].corr()
fig, ax = plt.subplots(figsize=(6*cm, 6*cm))
sns.heatmap(
    corrmat,
    annot=True,
    vmin=-1,
    vmax=1,
    center=0,
    cmap='coolwarm',
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.5}
)
ax.set_title('Correlation of pop. coupling')
fig.tight_layout()
my_savefig(fig=fig,figdir=figdir,filename='crossarea_corr_popcoupling_%s' % ses.session_id)
