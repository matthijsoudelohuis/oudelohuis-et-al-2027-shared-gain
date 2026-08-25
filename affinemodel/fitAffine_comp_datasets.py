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

from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from utils.params import params

figdir = os.path.join(params['figdir'],'affinemodel')

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_A_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_B_1_exampleses' + '.npz'),allow_pickle=True)
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_B_2_all' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load an example session: 
# data = np.load(os.path.join(os.getcwd(),'datasets','dataset_D_1_exampleses' + '.npz'),allow_pickle=True)
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_D_2_all' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_E_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% 
# sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='allfast',perc=50,recompute_poprate=True)
sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='area',perc=50,recompute_poprate=True)
# sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='area',perc=25,recompute_poprate=True)

#%% 
# sessions = fitAffine_GR_singleneuron_full(sessions,modelversion='allfast')

#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

ises = 0
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
scatter_alphabeta(ax,celldata)
# scatter_alphabeta(ax,celldata,
                #   xfield='aff_alpha_grfull',yfield='aff_beta_grfull')

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
