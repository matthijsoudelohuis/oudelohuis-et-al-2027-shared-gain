
#%% Import libs:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions
from loaddata.get_data_folder import get_local_drive
from utils.gain_lib import compute_pop_coupling
from utils.plot_lib import *
from utils.tuning import *

figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%% 
session_list        = np.array([['LPE11086_2024_01_05']])
session_list        = np.array([['LPE12223_2024_06_10']])

sessions,nSessions = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load proper data and compute average trial responses:                      
for ises in range(len(sessions)):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv')

#%% Add how neurons are coupled to the population rate: 
sessions = compute_pop_coupling(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% Fano factor per orientation, and averaged across orientations
sesidx  = 0
ses     = sessions[sesidx]

oris    = np.unique(ses.trialdata['Orientation'])
N       = ses.respmat.shape[0]
# For each neuron × orientation compute FF = Var / Mean, then average across oris.
# Using ddof=1 (unbiased variance) — important with small trial counts.
ff_per_ori = np.full((N, len(oris)), np.nan)
mean_per_ori = np.full((N, len(oris)), np.nan)
va_per_ori = np.full((N, len(oris)), np.nan)
for iori, ori in enumerate(oris):
    idx  = ses.trialdata['Orientation'] == ori
    # respmat = ses.respmat / ses.celldata['meanF'].to_numpy()[:, np.newaxis]
    # r    = respmat[:, idx]              # (N, n_trials_this_ori)
    r    = ses.respmat[:, idx]              # (N, n_trials_this_ori)
    # r = r/ses.celldata['meanF']
    mu   = r.mean(axis=1)
    # va   = r.var(axis=1, ddof=1)
    va   = r.std(axis=1, ddof=1)
    mean_per_ori[:, iori] = mu
    va_per_ori[:, iori] = va
    ff_per_ori[:, iori] = np.where(mu > 0, va / mu, np.nan)

fano_factor = np.nanmean(ff_per_ori, axis=1)   # (N,) — mean FF across orientations

#%% Plot for one neuron
lims = [3,300]

idx_N = np.random.choice(np.where(np.all((
    ses.celldata['gOSI']>0.5,
    ses.celldata['pop_coupling']>0.3),axis=0))[0])

fig, ax = plt.subplots(figsize=(4*cm, 4*cm))

# ax.scatter(mean_per_ori[idx_N,:].flatten(),
        # va_per_ori[idx_N,:].flatten(),s=.1,c='b',marker='.',edgecolors=None)
ax.scatter(mean_per_ori[idx_N,:].flatten(),va_per_ori[idx_N,:].flatten(),
           s=3,c='b',marker='.',linewidth=0)
ax.set_xlabel('mean')
ax.set_ylabel('variance')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
ax.plot(lims,lims,color='k',lw=.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
sns.despine(fig=fig,top=True,right=True,offset=2)

#%% 
# idx_N = np.random.choice(np.where(np.all((
    # ses.celldata['gOSI']>0.5,
    # ses.celldata['pop_coupling']>0.5),axis=0))[0])
idx_N = np.all((
    ses.celldata['gOSI']>0,
    # ses.celldata['gOSI']>0.5,
    # ses.celldata['pop_coupling']<0.2),axis=0)
    # ses.celldata['pop_coupling']>0.3),axis=0)
    ses.celldata['pop_coupling']>0),axis=0)

fig, ax = plt.subplots(figsize=(4*cm, 4*cm))
ax.scatter(mean_per_ori[idx_N,:].flatten(),va_per_ori[idx_N,:].flatten(),
           s=.3,c='b',marker='.',linewidth=0)
ax.set_xlabel('mean')
ax.set_ylabel('variance')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
ax.plot(lims,lims,color='k',lw=.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
# ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
sns.despine(fig=fig,top=True,right=True,offset=2)

#%% 
idx_N = np.all((
    # ses.celldata['gOSI']<0.4,
    ses.celldata['gOSI']>0,
    ses.celldata['pop_coupling']>0),axis=0)
    # ses.celldata['pop_coupling']<0.2),axis=0)

lims = [3,300]
fig, ax = plt.subplots(figsize=(5*cm, 4*cm))

x = mean_per_ori[idx_N, :].flatten()
y = va_per_ori[idx_N, :].flatten()
mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
x = x[mask]
y = y[mask]

edges = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 50)

sns.histplot(
    x=x,
    y=y,
    bins=(edges, edges),
    # cmap='magma',
    # cmap='Blues',
    cmap='rocket_r',
    cbar=True,
    cbar_kws={'label': 'count'},
    ax=ax,
    thresh=10
)

xlog = np.log10(x)
ylog = np.log10(y)
# Fit an exponential curve on logspace: log(y) = a + b * log(x) => y = 10^a * x^b
with np.errstate(over='ignore', invalid='ignore'):
    popt, _ = curve_fit(lambda t, a, b: a + b * t, xlog, ylog, maxfev=10000)
    a_fit, b_fit = popt
    xlog_fit = np.linspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
    ylog_fit = a_fit + b_fit * xlog_fit
    x_fit = 10**xlog_fit
    y_fit = 10**ylog_fit

ax.plot(x_fit, y_fit, color='k', lw=1.5, ls='--', label=f'power law fit: a={a_fit:.2f}, b={b_fit:.3f}')
ax.legend(loc='upper left', frameon=False, fontsize=5)

ax.set_xlabel('mean')
ax.set_ylabel('variance')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
ax.plot(lims,lims,color='k',lw=.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
# ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
sns.despine(fig=fig,top=True,right=True,offset=2)


#%% Plot: population coupling vs Fano factor
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
