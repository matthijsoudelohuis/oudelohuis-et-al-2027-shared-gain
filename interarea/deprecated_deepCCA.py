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
from sklearn.cross_decomposition import CCA

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.CCAlib import *

figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\Multiarea\\Nonlinear')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list)

#%% Load all GR sessions: 
# sessions,nSessions   = filter_sessions(protocols = 'GR')

#%%  Load data properly:        
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=calciumversion,keepraw=False)
    
#%% ########################### Compute tuning metrics: ###################################
# sessions = compute_tuning_wrapper(sessions)

#%%
# Author: Ronan Perry
# License: MIT

from mvlearn.embed import CCA, KMCCA, DCCA
from mvlearn.datasets import make_gaussian_mixture
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# GMM settings
n_samples = 200
centers = [[0, 0], [0, 0]]
covariances = 4*np.array([np.eye(2), np.eye(2)])
transforms = ['linear', 'poly', 'sin']

Xs_train_sets = []
Xs_test_sets = []
for transform in transforms:
    Xs_train, _ = make_gaussian_mixture(
        n_samples, centers, covariances, transform=transform, noise=0.25,
        noise_dims=2, random_state=41)
    Xs_test, _, latents = make_gaussian_mixture(
        n_samples, centers, covariances, transform=transform, noise=0.25,
        noise_dims=2, random_state=42, return_latents=True)

    Xs_train_sets.append(Xs_train)
    Xs_test_sets.append(Xs_test)


# Plotting parameters
labels = latents[:, 0]
cmap = matplotlib.colors.ListedColormap(
    sns.diverging_palette(240, 10, n=len(labels), center='light').as_hex())
cmap = 'coolwarm'

method_labels = \
    ['Raw Views', 'CCA', 'Polynomial KCCA', 'Gaussian KCCA', 'DCCA']
transform_labels = \
    ['Linear Transform', 'Polynomial Transform', 'Sinusoidal Transform']

input_size1 = Xs_train_sets[0][0].shape[1]
input_size2 = Xs_train_sets[0][1].shape[1]
outdim_size = min(Xs_train_sets[0][0].shape[1], 2)
layer_sizes1 = [256, 256, outdim_size]
layer_sizes2 = [256, 256, outdim_size]
methods = [
    CCA(regs=0.1, n_components=2),
    KMCCA(kernel='poly', regs=0.1, kernel_params={'degree': 2, 'coef0': 0.1},
          n_components=2),
    KMCCA(kernel='rbf', regs=0.1, kernel_params={'gamma': 1/4},
          n_components=2),
    DCCA(input_size1, input_size2, outdim_size, layer_sizes1, layer_sizes2,
         epoch_num=400)
]

fig, axes = plt.subplots(3 * 2, 5 * 2, figsize=(22, 12))
sns.set_context('notebook')

for r, transform in enumerate(transforms):
    axs = axes[2 * r:2 * r + 2, :2]
    for i, ax in enumerate(axs.flatten()):
        dim2 = int(i / 2)
        dim1 = i % 2
        ax.scatter(
            Xs_test_sets[r][0][:, dim1],
            Xs_test_sets[r][1][:, dim2],
            cmap=cmap,
            c=labels,
        )
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if dim1 == 0:
            ax.set_ylabel(f"View 2 Dim {dim2+1}", fontsize=14)
        if dim1 == 0 and dim2 == 0:
            ax.text(-0.4, -0.1, transform_labels[r], transform=ax.transAxes,
                    fontsize=22, rotation=90, verticalalignment='center')
        if dim2 == 1 and r == len(transforms)-1:
            ax.set_xlabel(f"View 1 Dim {dim1+1}", fontsize=14)
        if i == 0 and r == 0:
            ax.set_title(method_labels[r],
                         {'position': (1.11, 1), 'fontsize': 22})

    for c, method in enumerate(methods):
        axs = axes[2*r: 2*r+2, 2*c+2:2*c+4]
        Xs = method.fit(Xs_train_sets[r]).transform(Xs_test_sets[r])
        for i, ax in enumerate(axs.flatten()):
            dim2 = int(i / 2)
            dim1 = i % 2
            ax.scatter(
                Xs[0][:, dim1],
                Xs[1][:, dim2],
                cmap=cmap,
                c=labels,
            )
            if dim2 == 1 and r == len(transforms)-1:
                ax.set_xlabel(f"View 1 Dim {dim1+1}", fontsize=16)
            if i == 0 and r == 0:
                ax.set_title(method_labels[c + 1], {'position': (1.11, 1),
                             'fontsize': 22})
            ax.axis("equal")
            ax.set_xticks([], [])
            ax.set_yticks([], [])

plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)
plt.show()


#%%
from mvlearn.datasets import sample_joint_factor_model
from mvlearn.embed import CCA, MCCA, KMCCA
from mvlearn.plotting import crossviews_plot

#%% 
n_views = 3
n_samples = 1000
n_features = [10, 20, 30]
joint_rank = 3

# sample 3 views of data from a joint factor model
# m, noise_std control the difficulty of the problem
Xs, U_true, Ws_true = sample_joint_factor_model(
    n_views=n_views, n_samples=n_samples, n_features=n_features,
    joint_rank=joint_rank, m=5, noise_std=1, random_state=23,
    return_decomp=True)


# fit kernel MCCA with a linear kernel
kmcca = KMCCA(n_components=joint_rank, kernel='linear')

kmcca_scores = kmcca.fit_transform(Xs)
crossviews_plot(kmcca_scores[[0, 1]],
                title='KMCCA scores: linear kernel (first 2 views shown)',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})

print('Canonical Correlations:')
print(kmcca.canon_corrs(kmcca_scores))

#%%


#%% 
n_components        = 20
nsampleneurons      = 250
maxnoiselevel       = 20

areas               = np.array(['V1', 'PM'])
# weights_CCA         = np.full((len(areas),nsampleneurons,n_components,nSessions,nStim,nmodelfits),np.nan)
# popcoupling_CCA     = np.full((len(areas),nsampleneurons,nSessions,nStim,nmodelfits),np.nan)
do_cv_cca           = False

#%% Fit:
# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
ses                 = sessions[0]

idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                ses.celldata['noise_level']<maxnoiselevel,	
                ),axis=0))[0]
idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
            ses.celldata['noise_level']<maxnoiselevel,	
            ),axis=0))[0]

idx_areax_sub       = np.random.choice(idx_areax,np.min((np.sum(idx_areax),nsampleneurons)),replace=False)
idx_areay_sub       = np.random.choice(idx_areay[~np.isin(idx_areay,idx_areax_sub)],np.min((np.sum(idx_areay),nsampleneurons)),replace=False)

istim = 0
stim = np.unique(ses.trialdata['stimCond'])[istim] # loop over orientations

idx_T               = ses.trialdata['stimCond']==stim

X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T

X                   = zscore(X,axis=0)  #Z score activity for each neuron
Y                   = zscore(Y,axis=0)

Xs          = []
Xs.append(X)
Xs.append(Y)

#%% 
kernel = 'linear'

# fit kernel MCCA with a linear kernel
kmcca = KMCCA(n_components=joint_rank, kernel=kernel)

kmcca_scores = kmcca.fit_transform(Xs)

crossviews_plot(kmcca_scores[[0, 1]],
                title='KMCCA scores: ' + kernel + ' kernel (first 2 views shown)',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})

print('Canonical Correlations:')
print(kmcca.canon_corrs(kmcca_scores))





#%% Are the weights higher for coupled or uncoupled neuron?
n_components        = 20
nStim               = 16
nmodelfits          = 5
nsampleneurons      = 50
maxnoiselevel       = 20

areas               = np.array(['V1', 'PM'])
weights_CCA         = np.full((len(areas),nsampleneurons,n_components,nSessions,nStim,nmodelfits),np.nan)
popcoupling_CCA     = np.full((len(areas),nsampleneurons,nSessions,nStim,nmodelfits),np.nan)
do_cv_cca           = False

#%% Fit:
model_CCA           = CCA(n_components=n_components,scale = False, max_iter = 1000)

# for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
for ises,ses in enumerate(sessions):    # iterate over sessions
    
    idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                                    ses.celldata['noise_level']<maxnoiselevel,	
                                    ),axis=0))[0]
    idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
                                ses.celldata['noise_level']<maxnoiselevel,	
                                ),axis=0))[0]
    
    if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
        
        for imf in tqdm(range(nmodelfits),total=nmodelfits,desc='Fitting CCA model session %d/%d' % (ises+1,nSessions)):    # iterate over sessions

            idx_areax_sub       = np.random.choice(idx_areax,np.min((np.sum(idx_areax),nsampleneurons)),replace=False)
            idx_areay_sub       = np.random.choice(idx_areay[~np.isin(idx_areay,idx_areax_sub)],np.min((np.sum(idx_areay),nsampleneurons)),replace=False)
           
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations 
            # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
                idx_T               = ses.trialdata['stimCond']==stim
            
                X                   = sessions[ises].respmat[np.ix_(idx_areax_sub,idx_T)].T
                Y                   = sessions[ises].respmat[np.ix_(idx_areay_sub,idx_T)].T
                
                X                   = zscore(X,axis=0)  #Z score activity for each neuron
                Y                   = zscore(Y,axis=0)

                # Fit CCA MODEL:
                model_CCA.fit(X,Y)

                weights_CCA[0,:,:,ises,istim,imf]   = model_CCA.x_loadings_
                weights_CCA[1,:,:,ises,istim,imf]   = model_CCA.y_loadings_

                popcoupling_CCA[0,:,ises,istim,imf] = ses.celldata.loc[idx_areax_sub,'pop_coupling']
                popcoupling_CCA[1,:,ises,istim,imf] = ses.celldata.loc[idx_areay_sub,'pop_coupling']


#%% 
 #####   #####     #       ######  ####### ######   #####  
#     # #     #   # #      #     # #     # #     # #     # 
#       #        #   #     #     # #     # #     # #       
#       #       #     #    ######  #     # ######   #####  
#       #       #######    #       #     # #             # 
#     # #     # #     #    #       #     # #       #     # 
 #####   #####  #     #    #       ####### #        #####  

#%% Are the CCA correlations higher for choristers or soloists across areas?
kFold               = 5
n_components        = 20
nStim               = 16
nmodelfits          = 5
nsampleneurons      = 50
# nsampleneurons      = 100
maxnoiselevel       = 20
nbins_popcoupling   = 5
couplinglabels      = ['0-20%','20-40%','40-60%','60-80%','80-100%']
# couplinglabels      = ['Soloists','Choristers']
areas               = np.array(['V1', 'PM'])
CCA_corrtest        = np.full((nbins_popcoupling,n_components,nSessions,nStim),np.nan)
prePCA              = 25

#%% Fit:
for ises,ses in tqdm(enumerate(sessions),total=nSessions,desc='Fitting CCA model'):    # iterate over sessions
    binedges_popcoupling   = np.percentile(ses.celldata['pop_coupling'],np.linspace(0,100,nbins_popcoupling+1))

    for ibin in range(nbins_popcoupling):
        idx_areax           = np.where(np.all((ses.celldata['roi_name']==areas[0],
                                        ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
                                        ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
        idx_areay           = np.where(np.all((ses.celldata['roi_name']==areas[1],
                                        ses.celldata['pop_coupling']>binedges_popcoupling[ibin],
                                        ses.celldata['pop_coupling']<binedges_popcoupling[ibin+1],
                                        ses.celldata['noise_level']<maxnoiselevel),axis=0))[0]
            
        if len(idx_areax)>nsampleneurons and len(idx_areay)>nsampleneurons:
            # for istim,stim in tqdm(enumerate(np.unique(ses.trialdata['stimCond'])),total=nStim,desc='Fitting CCA model'):    # iterate over sessions
            for istim,stim in enumerate(np.unique(ses.trialdata['stimCond'])): # loop over orientations
                idx_T               = ses.trialdata['stimCond']==stim
            
                X                   = sessions[ises].respmat[np.ix_(idx_areax,idx_T)].T
                Y                   = sessions[ises].respmat[np.ix_(idx_areay,idx_T)].T
                
                [g,_] = CCA_subsample(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=prePCA,
                    n_components=np.min([n_components,nsampleneurons]))
                CCA_corrtest[ibin,:,ises,istim] = g

                # [g,_] = CCA_subsample_it(X,Y,nN=nsampleneurons,resamples=nmodelfits,kFold=kFold,prePCA=prePCA,
                #                          n_components=np.min([n_components,nsampleneurons]))
                # CCA_corrtest[ibin,:,ises,istim] = g

# CCA_validate_iterative(X,Y,prePCA=prePCA,n_components=n_components)

#%%
clrs_couplingbins = sns.color_palette('magma',nbins_popcoupling)
fig, axes = plt.subplots(1,1,figsize=(3,3))

ax = axes
handles = []
# for iapl, arealabelpair in enumerate(ncouplingpairs):
for icpl in range(nbins_popcoupling):
    # ax.plot(np.arange(n_components),np.nanmean(CCA_corrtest[iapl,:,:,:],axis=(1,2)),
            # color=clrs_arealabelpairs[iapl],linewidth=2)
    iapldata = CCA_corrtest[icpl,:,:,:].reshape(n_components,-1)
    handles.append(shaded_error(x=np.arange(n_components),
                                y=iapldata.T,
                                # error='sem',color='k',alpha=0.3,ax=ax))
                                error='sem',color=clrs_couplingbins[icpl],alpha=0.3,ax=ax))
ax.set_xticks(np.arange(0,n_components+5,5))
ax.set_xticklabels(np.arange(0,n_components+5,5)+1)
ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(CCA_corrtest,axis=(2,3))),1)])
ax.set_yticks([0,ax.get_ylim()[1]/2,ax.get_ylim()[1]])
ax.set_xlabel('CCA Dimension')
ax.set_ylabel('Correlation')
ax.legend(handles,couplinglabels,loc='upper right',reverse=True,fontsize=7,frameon=True,title='pop. coupling')
sns.despine(top=True,right=True,offset=1,trim=True)
my_savefig(fig,figdir,'CCA_V1PM_PopCoupling_BothAreas_testcorr_%dsessions' % (nSessions),formats=['png'])
