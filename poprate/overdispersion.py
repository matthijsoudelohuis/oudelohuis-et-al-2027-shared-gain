
#%% Import libs:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions
from utils.gain_lib import compute_pop_coupling
from utils.plot_lib import *
from utils.tuning import *
from utils.params import params

figdir = os.path.join(params['figdir'],'overdispersion')

#%% Load a dataset:
dataset_name = 'dataset_A_1_exampleses'
# dataset_name = 'dataset_B_1_exampleses'
# dataset_name = 'dataset_B_2_all'
# dataset_name = 'dataset_D_2_all'
# dataset_name = 'dataset_E_1_exampleses'

data = np.load(os.path.join(os.getcwd(),'datasets',dataset_name + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load a synthetic dataset:
dataset_name = 'synthetic_affine'
sessions = [generate_affine_data(gain_level=1,offset_level=1,noise_level=1)]

#%% 
def scatter_variance_mean(ax,sessions,varmetric='variance'):
    celldata = pd.concat([ses.celldata for ses in sessions])

    oris    = np.unique(sessions[0].trialdata['Orientation'])

    N       = len(celldata)
    # For each neuron × orientation compute FF = Var / Mean, then average across oris.
    # Using ddof=1 (unbiased variance) — important with small trial counts.
    mean_per_ori    = np.full((N, len(oris)), np.nan)
    va_per_ori      = np.full((N, len(oris)), np.nan)
    ff_per_ori      = np.full((N, len(oris)), np.nan)

    for ises,ses in enumerate(sessions):
        idx_ses        = celldata['session_id']==ses.session_id
        for iori, ori in enumerate(oris):
            idx  = ses.trialdata['Orientation'] == ori
            # respmat = ses.respmat / ses.celldata['meanF'].to_numpy()[:, np.newaxis]
            # r    = respmat[:, idx]              # (N, n_trials_this_ori)
            r    = ses.respmat[:, idx]              # (N, n_trials_this_ori)
            # r = r/ses.celldata['meanF']
            mu   = r.mean(axis=1)
            if varmetric=='variance':
                va   = r.var(axis=1, ddof=1)
            elif varmetric=='std':
                va   = r.std(axis=1, ddof=1)
            mean_per_ori[idx_ses, iori] = mu
            va_per_ori[idx_ses, iori] = va
            ff_per_ori[idx_ses, iori] = np.where(mu > 0, va / mu, np.nan)

    fano_factor = np.nanmean(ff_per_ori, axis=1)   # (N,) — mean FF across orientations

    ax.scatter(mean_per_ori.flatten(),va_per_ori.flatten(),
            s=.3,c='b',marker='.',linewidth=0)
    ax.set_xlabel('mean')
    ax.set_ylabel('variance')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
    ax.plot([0,1e9],[0,1e9],color='k',lw=.5,linestyle=':')
    ax.set_xlim(np.percentile(mean_per_ori,[1,100]))
    ax.set_ylim(np.percentile(va_per_ori,[1,100]))
    # ax.plot([0,ax.get_xlim()[1]],[0,ax.get_ylim()[1]],color='k',lw=1)
    sns.despine(ax=ax,top=True,right=True,offset=2)
    return fano_factor

def scatter_fano_popcoupling(ax,celldata):
    # Plot: population coupling vs Fano factor
    sns.regplot(data=celldata, x='pop_coupling', y='fano_factor', ax=ax,
                scatter_kws={'s': .5, 'alpha': 1, 'color': 'blue', 'rasterized': True},
                line_kws={'color': 'k', 'lw': 1})
    ax.text(0.6,0.8,'N = %d\nneurons' % (N),transform=ax.transAxes)
    r,pval = pearsonr(celldata['pop_coupling'], celldata['fano_factor'])[:2]
    ax.text(0.1,0.8,'r=%.2f\np=%2.2g' % (r,pval),transform=ax.transAxes, color='k')
    sns.despine(ax=ax, top=True, right=True, offset=2)

#%%
fig, ax = plt.subplots(figsize=(4*cm, 3.7*cm))

fano_factor = scatter_variance_mean(ax,sessions,varmetric='variance')
celldata = pd.concat([ses.celldata for ses in sessions])
celldata['fano_factor'] = fano_factor
my_savefig(fig=fig,figdir=figdir,filename='scatter_variance_mean_%s' % (dataset_name))

#%% 
fig, ax = plt.subplots(figsize=(4*cm, 3.7*cm))
scatter_fano_popcoupling(ax,celldata)
my_savefig(fig=fig,figdir=figdir,filename='fanofactor_popcoupling_%s' % (dataset_name))

#%% Load a synthetic dataset:
dataset_name = 'synthetic_affine'
# sessions = [generate_affine_data(tuning_level=2,gain_level=4,offset_level=0.0,noise_level=0.1,poisson_level=1)]
sessions = [generate_affine_data(tuning_level=2,gain_level=4,offset_level=0,noise_level=0.15,poisson_level=0)]
# sessions = [generate_nonlin_data(nonlin='Linear',tuning_level=1,popmodulation_level=5,noise_std=0.5)]
sessions = [generate_nonlin_data(nonlin='Exp',tuning_level=1,popmodulation_level=5,noise_std=.6)]
# sessions = [generate_nonlin_data(nonlin='Sigmoid',tuning_level=1,popmodulation_level=.1,noise_std=0.05)]
# sessions = [generate_nonlin_data(nonlin='Sigmoid',tuning_level=1,popmodulation_level=50,noise_std=0.25)]
fig, ax = plt.subplots(figsize=(4*cm, 3.7*cm))
fano_factor = scatter_variance_mean(ax,sessions,varmetric='variance')
# celldata = pd.concat([ses.celldata for ses in sessions])
# celldata['fano_factor'] = fano_factor
# my_savefig(fig=fig,figdir=figdir,filename='scatter_variance_mean_%s' % (dataset_name))


#%% Plot for one neuron
lims = [3,300]
lims = np.nanpercentile(mean_per_ori,[1,99])

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

lims = np.nanpercentile(mean_per_ori,[1,99])
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

