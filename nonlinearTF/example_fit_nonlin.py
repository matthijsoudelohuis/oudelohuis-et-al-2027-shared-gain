#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from utils.gain_lib import * 
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import *
from utils.nonlin_lib import *
from utils.params import params

figdir = os.path.join(params['figdir'],'nonlinearTF')

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_A_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% ###########################################################################
# NONLINEAR TRANSFER FUNCTION FITTING PIPELINE
# Model: r(t) = f( θ_k(t) + γ · P(t) + b )
#   θ_k  : stimulus drive — one free parameter per orientation (16)
#   γ    : population-rate scaling (additive input, 1 param)
#   b    : input bias (1 param)
#   f(·) : nonlinearity (with model-specific free parameters)
###########################################################################

#%% Show nonlinearities at p0 initialization
x = np.linspace(-0.5, 1.25, 300)
fig, axes = plt.subplots(1, nNL, figsize=(nNL * 2.5*cm, 3.7*cm), sharey=True)
for i, (name, nl_func, n_shape, p0_shape, _) in enumerate(NL_CONFIGS):
    ax = axes[i]
    y  = nl_func(x, *p0_shape) if n_shape else nl_func(x)
    ax.plot(x, y, color=clrs_nl[i], lw=2)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('u  (θ_k + γ·P + b)')
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    if i == 0:
        ax.set_ylabel('f(u)')
    if p0_shape:
        ax.text(0.05, 0.95, ', '.join([f'{v}' for v in p0_shape]),
                transform=ax.transAxes, fontsize=7, va='top', color='gray')

sns.despine(trim=True, offset=3)

plt.suptitle('Nonlinearities at p0 initialization', fontsize=10, y=1.02)
plt.tight_layout()
# my_savefig(fig, figdir, 'NL_p0_shapes', formats=['png'])

#%%
# sessions = comp_poprate(sessions,version='radius_500')

#%% Pick example neuron: well-tuned with moderate–high pop coupling
ises     = 0
ses      = sessions[ises]

ustim    = np.unique(ses.trialdata['Orientation'])
stim_ids = np.searchsorted(ustim, ses.trialdata['Orientation'].to_numpy())
nstim    = len(ustim)

idx_good = np.where(
    (ses.celldata['gOSI']           > 0.5) &
    # (ses.celldata['gOSI']           <0.2) &
    (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 70))
    # (ses.celldata['pop_coupling']  > np.percentile(ses.celldata['pop_coupling'], 50)) &
    # (ses.celldata['noise_level']   < 20)
    )[0]

# sigmoidaldiff = ses.celldata['R2Sigmoid'] - ses.celldata['R2Power-law (p=2)']
# idx_good = np.where(sigmoidaldiff > np.nanpercentile(sigmoidaldiff, 80))[0]
ex_iN    = np.random.choice(idx_good)
# ex_iN = 0
resp_ex  = ses.respmat[ex_iN, :]
# Normalise responses to [0, 1]
r_min     = resp_ex.min()
r_max     = resp_ex.max()
# r_max     = np.percentile(resp_ex, 99)
resp_ex = (resp_ex - r_min) / max(r_max - r_min, 1e-8)
# resp_ex = zscore(resp_ex)
poprate = ses.popratemat[ex_iN,:] 

ex_cid   = ses.celldata['cell_id'].iloc[ex_iN]
print(f'Example: {ex_cid}  OSI={ses.celldata["gOSI"].iloc[ex_iN]:.2f}  '
      f'pop_coupling={ses.celldata["pop_coupling"].iloc[ex_iN]:.2f}')

results_ex = fit_nl_models(resp_ex, stim_ids, poprate, NL_CONFIGS)

fig = diagnostic_nonlinfit(results_ex)

# my_savefig(fig, figdir, f'NLfit_diagnostics_{ex_cid}', formats=['png'])

#%%
[sessions, theta_arr, nlpar_arr, ses_idx_arr] = fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS)
#  fit_nl_models(sessions, nl_configs=NL_CONFIGS, verbose=False):

#%% Plot R² distributions across models and neurons
celldata = pd.concat([ses.celldata for ses in sessions])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
hist_perf_nonlinfits(ax,celldata)
ax = axes[1]
violin_perf_nonlinfits(ax,celldata)
plt.tight_layout()
# my_savefig(fig, figdir, f'NLfit_R2_distributions_{nSessions}sessions', formats=['png'])

#%%
def hist_perf_nonlinfits(ax,celldata):
    bw_adjust = 0.25
    for i, name in enumerate(nl_names):
        vals = celldata['R2' + name].dropna().values
        vals = vals[vals > -1]
        sns.kdeplot(vals, ax=ax, color=clrs_nl[i], label=name, 
                    bw_adjust=bw_adjust, clip=[0,1],fill=False, lw=1)
    ax.set_xlabel('R²')
    ax.set_ylabel('Density')
    ax.set_title('R² distribution across neurons')
    ax.legend(frameon=False)
    sns.despine(ax=ax, trim=True, offset=3)

def violin_perf_nonlinfits(ax,celldata):
    bw_adjust = 0.25
    r2_df = celldata[['R2' + name for name in nl_names]].dropna()
    r2_long = (r2_df.clip(lower=-1)
                .melt(var_name='Model', value_name='R²')
                .dropna())
    r2_long['Model'] = r2_long['Model'].str.replace('R2', '')
    sns.violinplot(data=r2_long, x='Model', y='R²', hue='Model',palette=clrs_nl, ax=ax,
                    bw_adjust=bw_adjust,inner='quartile', cut=0, order=nl_names)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_title('R² distribution (violin)')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    sns.despine(ax=ax, trim=True, offset=3)

def perf_nonlinfits_bestvslinear(ax,celldata):
    fields = ['R2'+name for name in nl_names]
    best_field = celldata[fields].mean().idxmax()
    best_name = best_field.replace('R2', '', 1)
    xdata = celldata['R2Linear'].dropna().clip(lower=-1)
    ydata = celldata[best_field].dropna().clip(lower=-1)

    ax.scatter(xdata,ydata,marker='.',s=5,color='k')
    ax.set_ylabel(f'{best_name} R²')
    ax.set_title(f'Best TF vs. Linear: {best_name}')
    ax.plot([0,1],[0,1], color='k', lw=0.5, ls=':')
    sns.despine(ax=ax, trim=True, offset=3)

#%% Scatter: pop_coupling vs fitted gamma across models
from utils.corr_lib import filter_sharednan
idx_N = np.all((
    # celldata['noise_level']<20,
                celldata['gOSI']>0,
                # celldata['pop_coupling']>np.percentile(celldata['pop_coupling'],50),
                ),axis=0)
ncols = nNL
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3), sharex=True, sharey=False)

for i, name in enumerate(nl_names):
    ax     = axes[i]
    y  = celldata['Gamma' + name][idx_N]
    x  = celldata['pop_coupling'][idx_N]
    # pc     = pop_coupling_all
    x,y = filter_sharednan(x,y)
    # mask = np.isfinite(gamma) & np.isfinite(pc)
    # x, y = pc[mask], gamma[mask]

    ax.scatter(x, y, s=3, alpha=0.3, color=clrs_nl[i], rasterized=True)

    slope, intercept, r, p, _ = linregress(x, y)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color='k', lw=1.5, ls='--')

    ax.set_xlabel('Pop. coupling', fontsize=9)
    if i == 0:
        ax.set_ylabel('Fitted γ', fontsize=9)
    ax.set_title(name, fontsize=9)
    ax.set_xlim([-0.2,0.6])
    ax.set_ylim(np.percentile(y, [0, 98]))
    ax.text(0.05, 0.93, f'r={r:.2f}, p={p:.1e}',
            transform=ax.transAxes, fontsize=7, va='top')
    sns.despine(ax=ax, trim=True, offset=3)

plt.suptitle('Population coupling vs fitted γ  (pop-rate scaling)', fontsize=10, y=1.02)
plt.tight_layout()
# my_savefig(fig, figdir, f'PopCoupling_vs_gamma_{nSessions}sessions', formats=['png'])

#%% Scatter of linear vs. best model R2
fig, axes = plt.subplots(1, len(nl_names)-1, figsize=((len(nl_names)-1) * 2, 2),
                         sharex=True, sharey=True)
for i, name in enumerate(nl_names[1:]):
    ax = axes[i]
    xdata = celldata['R2Linear'][idx_N]
    ydata = celldata['R2' + name][idx_N]
    ax.scatter(xdata, ydata, s=5, alpha=0.5, color=clrs_nl[i+1], rasterized=True)
    add_paired_ttest_results(ax, xdata, ydata, pos=[0.2,0.9])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Linear R²', fontsize=9)
    ax.set_ylabel(f'{name} R²', fontsize=9)
    ax.plot([0,1], [0,1], color='k', lw=0.5, ls=':')
sns.despine(trim=True, offset=3)

plt.tight_layout()
# my_savefig(fig, figdir, f'NLfit_R2_scatter_{nSessions}sessions', formats=['png'])


#%%

#%% Load an example session: 
# data = np.load(os.path.join(os.getcwd(),'datasets','dataset_B_1_exampleses' + '.npz'),allow_pickle=True)
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_D_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Pick example neuron: well-tuned with moderate–high pop coupling
ises     = 0
ses      = sessions[ises]

ustim    = np.unique(ses.trialdata['Orientation'])
stim_ids = np.searchsorted(ustim, ses.trialdata['Orientation'].to_numpy())
nstim    = len(ustim)

idx_good = np.where(np.all((
    ses.celldata['gOSI']> np.percentile(ses.celldata['gOSI'], 50),
    ses.celldata['pop_coupling'] > np.percentile(ses.celldata['pop_coupling'],50)
    ),axis=0))[0]

# sigmoidaldiff = ses.celldata['R2Sigmoid'] - ses.celldata['R2Power-law (p=2)']
# idx_good = np.where(sigmoidaldiff > np.nanpercentile(sigmoidaldiff, 80))[0]
ex_iN    = np.random.choice(idx_good)
resp_ex  = ses.respmat[ex_iN, :]
# Normalise responses to [0, 1]
r_min     = resp_ex.min()
r_max     = resp_ex.max()
resp_ex = (resp_ex - r_min) / max(r_max - r_min, 1e-8)
poprate = ses.popratemat[ex_iN,:] 

results_ex = fit_nl_models(resp_ex, stim_ids, poprate, NL_CONFIGS)

fig = diagnostic_nonlinfit(results_ex)

#%%
[sessions, theta_arr, nlpar_arr, ses_idx_arr] = fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS)
#  fit_nl_models(sessions, nl_configs=NL_CONFIGS, verbose=False):

#%% Plot R² distributions across models and neurons
celldata = pd.concat([ses.celldata for ses in sessions])
fig, axes = plt.subplots(1, 3, figsize=(12*cm, 4*cm))
ax = axes[0]
hist_perf_nonlinfits(ax,celldata)
ax = axes[1]
violin_perf_nonlinfits(ax,celldata)
ax = axes[2]
perf_nonlinfits_bestvslinear(ax,celldata)
plt.tight_layout()

