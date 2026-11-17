#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress, spearmanr, rankdata
import statsmodels.formula.api as smf

from sharedgain.utils.tuning import compute_tuning_wrapper
from sharedgain.utils.gain_lib import * 
from sharedgain.utils.pair_lib import compute_pairwise_anatomical_distance
from sharedgain.utils.plot_lib import * #get all the fixed color schemes
from sharedgain.utils.params import params
from sharedgain.utils.nonlin_lib import * 
from sharedgain.paths import pipeline_dir

figdir = pipeline_dir('05b_fitAffine_comp_datasets','out')

#%% Load a dataset:
# dataset_name = 'dataset_A_1_exampleses'
# dataset_name = 'dataset_B_1_exampleses'
# dataset_name = 'dataset_B_2_all'
dataset_name = 'dataset_D_2_all'
# dataset_name = 'dataset_E_1_exampleses'

data = np.load(os.path.join(pipeline_dir('00a_create_datasets','store'),dataset_name + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load a synthetic dataset:
dataset_name = 'synthetic_affine'
sessions = [generate_affine_data(gain_level=1,offset_level=1,noise_level=1)]

#%% 
sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='allfast',perc=50,recompute_poprate=True)
# sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='area',perc=50,recompute_poprate=True)
# sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='area',perc=25,recompute_poprate=True)

#%% 
# sessions = fitAffine_GR_singleneuron_full(sessions,modelversion='allfast')

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

ises = 0
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
scatter_alphabeta(ax,celldata)
# my_savefig(fig=fig,figdir=figdir,filename='scatter_alpha_beta_%s' % (dataset_name))

#%%
ises = 0
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
# alpha_rank = rankdata(celldata['aff_alpha_grfull'])
# beta_rank = rankdata(celldata['aff_beta_grfull'])
alpha_rank = rankdata(celldata['aff_alpha_grsplit'])
beta_rank = rankdata(celldata['aff_beta_grsplit'])
rho, pval = spearmanr(celldata['aff_alpha_grsplit'], celldata['aff_beta_grsplit'])
sns.scatterplot(x=alpha_rank,y=beta_rank,color='green',ax=ax,marker='.',s=8)
sns.regplot(x=alpha_rank,y=beta_rank,
            color='k',line_kws={'linewidth': 1},scatter=False)
ax.set_xlabel('alpha rank')
ax.set_ylabel('beta rank')
ax.set_title(f'Spearman $\\rho$ = {rho:.2f}')
sns.despine(ax=ax,top=True,right=True,offset=2)

#%% Show different affine fits
sessions = [generate_affine_data(tuning_level=1,gain_level=.5,offset_level=0,noise_level=.1,poisson_level=0)]
sessions = [generate_affine_data(tuning_level=1,gain_level=0,offset_level=0.5,noise_level=.1,poisson_level=0)]
# sessions = [generate_nonlin_data(nonlin='Linear',tuning_level=1,popmodulation_level=5,noise_std=.15,drive_mean=10)]

# sessions = [generate_nonlin_data(nonlin='Exp',tuning_level=1,popmodulation_level=.5,noise_std=.5,
#                                  drive_mean=-0.25)]
 
sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='allfast',perc=50,recompute_poprate=True)
sessions = compute_pop_coupling(sessions,version='allfast')
sessions = compute_tuning_wrapper(sessions)
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

fig,ax = plt.subplots(1,1,figsize=(4*cm,4*cm))
scatter_alphabeta(ax,celldata,hue='pop_coupling')
# scatter_alphabeta(ax,celldata,hue='gOSI')
# scatter_alphabeta(ax,celldata,hue='roi_name')
# my_savefig(fig=fig,figdir=figdir,filename='pop_geometry_synthetic_affine_onlynoise')

#%% Show different geometries:
sessions = [generate_affine_data(tuning_level=1,gain_level=5,offset_level=0,noise_level=.1,poisson_level=1)]

sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='allfast',perc=50,recompute_poprate=True)
sessions = compute_pop_coupling(sessions,version='allfast')
sessions = compute_tuning_wrapper(sessions)
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

fig,ax = plt.subplots(1,1,figsize=(4*cm,4*cm))
scatter_alphabeta(ax,celldata,hue='pop_coupling')
