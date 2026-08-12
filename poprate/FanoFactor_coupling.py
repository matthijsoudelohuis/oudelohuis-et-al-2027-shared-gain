
#%% Import libs:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions
from loaddata.get_data_folder import get_local_drive
from utils.gain_lib import compute_pop_coupling

figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%%
cm = 1/2.54  # centimeters in inches

#%% 
session_list        = np.array([['LPE11086_2024_01_05']])
# session_list        = np.array([['LPE12223_2024_06_10']])

sessions = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load proper data and compute average trial responses:                      
for ises in range(len(sessions)):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)


#%% Fano factor per orientation, averaged across orientations
sesidx  = 0
ses     = sessions[sesidx]

oris    = np.unique(ses.trialdata['Orientation'])
N       = ses.respmat.shape[0]
# For each neuron × orientation compute FF = Var / Mean, then average across oris.
# Using ddof=1 (unbiased variance) — important with small trial counts.
ff_per_ori = np.full((N, len(oris)), np.nan)
for iori, ori in enumerate(oris):
    idx  = ses.trialdata['Orientation'] == ori
    r    = ses.respmat[:, idx]              # (N, n_trials_this_ori)
    mu   = r.mean(axis=1)
    # va   = r.var(axis=1, ddof=1)
    va   = r.std(axis=1, ddof=1)
    ff_per_ori[:, iori] = np.where(mu > 0, va / mu, np.nan)

fano_factor = np.nanmean(ff_per_ori, axis=1)   # (N,) — mean FF across orientations

#Plot: population coupling vs Fano factor
pop_coupling = ses.celldata['pop_coupling'].values

df = pd.DataFrame({'Population coupling': pop_coupling,
                   'SD/Mean': fano_factor}).dropna()

fig, ax = plt.subplots(figsize=(6*cm, 6*cm))
sns.regplot(data=df, x='Population coupling', y='SD/Mean', ax=ax,
            scatter_kws={'s': 2, 'alpha': 0.2, 'color': 'gray', 'rasterized': True},
            line_kws={'color': 'r', 'lw': 1.5})
ax.set_title(ses.session_id, fontsize=9)
ax.text(0.6,0.8,'N = %d\nneurons' % (N),transform=ax.transAxes,fontsize=7)
ax.text(0.6,0.6,'r=%.2f' % np.corrcoef(pop_coupling, fano_factor)[0,1],
        transform=ax.transAxes, fontsize=7,color='r')
sns.despine(fig=fig, top=True, right=True, offset=3)
fig.tight_layout()
plt.show()
