# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023-2025, Champalimaud Center
"""

#%% 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore
from scipy import stats
from sklearn.metrics import r2_score
from statannotations.Annotator import Annotator

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.imagelib import * 
from utils.tuning import *
from utils.corr_lib import compute_signal_noise_correlation
from utils.gain_lib import *
from utils.rf_lib import *
from utils.regress_lib import *

figdir = os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\SharedGain\\')

#%% ################################################
session_list        = np.array([['LPE11086_2023_12_16']])
session_list        = np.array([['LPE13959_2025_02_24']])

#%% Load sessions lazy: 
sessions,nSessions   = filter_sessions(protocols = ['IM'],only_session_id=session_list)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_lab_cells_V1=50,min_lab_cells_PM=50)
# sessions,nSessions   = filter_sessions(protocols = ['IM'],min_cells=1)

#%%   Load proper data and compute average trial responses:                      
for ises in range(nSessions):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

#%% ### Load the natural images:
# natimgdata = load_natural_images(onlyright=False)
natimgdata = load_natural_images(onlyright=True)

#%% Compute tuning metrics:
for ses in sessions: 
    ses.respmean,imageids = mean_resp_image(ses)

#%% Compute tuning metrics of natural images:
for ses in tqdm(sessions,desc='Computing tuning metrics for each session'): 
    ses.celldata['tuning_SNR']                          = compute_tuning_SNR(ses)
    ses.celldata['corr_half'],ses.celldata['rel_half']  = compute_splithalf_reliability(ses)
    ses.celldata['sparseness']          = compute_sparseness(ses.respmat)
    ses.celldata['selectivity_index']   = compute_selectivity_index(ses.respmat)
    ses.celldata['fano_factor']         = compute_fano_factor(ses.respmat)
    ses.celldata['gini_coefficient']    = compute_gini_coefficient(ses.respmat)

#%%



#%% On the trial to trial response: Ridge regression to get RF
nsub    = 3 #without subsampling really slow, i.e. nsub=1
lam     = 0.05

for ises, ses in enumerate(sessions):
    print(ses.session_id)
    resp    = ses.respmat.T

    K,N     = np.shape(resp)

    #normalize the response for each neuron to the maximum:
    resp        = zscore(resp, axis=0)

    # dividing by poprate:
    # resp = resp / np.mean(resp, axis=0, keepdims=True)

    IMdata      = natimgdata[:,:,ses.trialdata['ImageNumber']]

    # cRF,Y_hat = lowrank_RF_cv(resp, IMdata,lam=0.05,nranks=100,nsub=nsub)
    ses.cRF,Y_hat   = linear_RF_cv(resp, IMdata, lam=lam, nsub=nsub)

    RF_R2 = r2_score(resp,Y_hat,multioutput='raw_values')
    ses.celldata['RF_R2'] = RF_R2
    print('RF R2: %0.2f' % (RF_R2.mean()))

    #Compute pairwise correlations matrix of the cRF:
    cRF_reshape             = np.reshape(ses.cRF, (np.shape(ses.cRF)[0]*np.shape(ses.cRF)[1],np.shape(ses.cRF)[2]))
    ses.RF_corrmat  = np.corrcoef(cRF_reshape.T)

    np.fill_diagonal(ses.RF_corrmat,np.nan)

    ses.resp_corr = np.corrcoef(ses.respmat)
    # sessions[sesidx].resp_corr = np.corrcoef(resp)
    np.fill_diagonal(ses.resp_corr,np.nan)

    ses.resid_corr = np.corrcoef(resp.T - Y_hat.T)
    np.fill_diagonal(ses.resid_corr,np.nan)

#%% Show example neurons for different populations:
arealabels      = ['V1','PM']
narealabels     = len(arealabels)
nN              = 6 #number of example neurons to show
sesidx          = 0

fig,axes = plt.subplots(nN,narealabels,figsize=(3*narealabels,1*nN))
for ial,arealabel in enumerate(arealabels):
    idx_neurons = np.where(np.all((sessions[sesidx].celldata['roi_name']==arealabel,
                                   ),axis=0))[0]
    idx_neurons = idx_neurons[np.argsort(-sessions[sesidx].celldata['RF_R2'][idx_neurons])][:nN]

    for i,iN in enumerate(idx_neurons):
        ax = axes[i,ial]
        lim = np.max(np.abs(sessions[sesidx].cRF[:,:,iN]))*1.2
        ax.imshow(sessions[sesidx].cRF[:,:,iN],cmap='bwr',vmin=-lim,
                            vmax=lim)
        # ax.plot(sessions[sesidx].celldata['rf_az_RRR'][iN]/nsub,
        #         sessions[sesidx].celldata['rf_el_RRR'][iN]/nsub,'k+',markersize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.set_title(arealabel + sessions[sesidx].celldata['cell_id'][iN],fontsize=8)
# fig.savefig(os.path.join(figdir,'ReducedRank_FitRF_example_neurons_arealabel_%s.png' % sessions[sesidx].sessiondata['session_id'][0]),format='png',dpi=300,bbox_inches='tight')

#%%
sesidx = 0
plt.hist(sessions[sesidx].celldata['RF_R2'], bins=100, color='blue', alpha=0.5)


#%% Show the difference in the distribution of the responses for predicted vs true data:
plt.hist(Y_hat.flatten(), bins=100, color='red', alpha=0.5)
plt.hist(resp.flatten(), bins=100, color='blue', alpha=0.5)
# plt.savefig(os.path.join(figdir,'LowRank_RF_FittedResponses_%s.png' % (sessions[sesidx].sessiondata['session_id'][0])),format='png',dpi=300,bbox_inches='tight')

#%% Fit the cRF with a 2D gaussian:
# sessions[sesidx].celldata = fit_2dgauss_cRF(ses.cRF, nsub=nsub,celldata=sessions[sesidx].celldata)

#%% Compute EV of the cRF:
print('Fraction of variance explained: %1.3f' % EV(resp,Y_hat))

sessions[sesidx].celldata['RF_R2'] = r2_score(resp,Y_hat,multioutput='raw_values')
print('Average per neuron fraction of variance explained: %1.3f' % np.mean(sessions[sesidx].celldata['RF_R2']))

#%% Sanity check:
# get five entries that have high values, not along the diagonal, show RF for the two neurons
corrmat_triu = np.triu(sessions[sesidx].RF_corrmat,k=1)

idx_neurons = sessions[sesidx].celldata['RF_R2']<0.025
corrmat_triu[idx_neurons,:] = 0 #set poor neurons to zero

idx_corr = np.unravel_index(np.argsort(corrmat_triu, axis=None)[-5:], corrmat_triu.shape)

fig,ax = plt.subplots(5,2,figsize=(5,4),sharex=True,sharey=True)
# for i,idx in enumerate(idx_corr):
for i,idx in enumerate(zip(idx_corr[0],idx_corr[1])):
    lim = np.max(np.abs(sessions[sesidx].cRF[:,:,idx[0]]))*1.3
    ax[i,0].imshow(sessions[sesidx].cRF[:,:,idx[0]],cmap='bwr',vmin=-lim,vmax=lim)
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    ax[i,0].set_axis_off()
    ax[i,0].set_title(sessions[sesidx].celldata['cell_id'][idx[0]],fontsize=8)
    ax[i,1].imshow(sessions[sesidx].cRF[:,:,idx[1]],cmap='bwr',vmin=-lim,vmax=lim)
    ax[i,1].set_xticks([])
    ax[i,1].set_yticks([])
    ax[i,1].set_axis_off()
    ax[i,1].set_title(sessions[sesidx].celldata['cell_id'][idx[1]],fontsize=8)
    ax[i,0].text(0.9,0.6,'RF corr: %.2f' % corrmat_triu[idx],fontsize=8,transform=ax[i,0].transAxes)
fig.tight_layout()
# fig.savefig(os.path.join(figdir,'FitRF_linear_example_RF_corr_%s.png' % sessions[sesidx].sessiondata['session_id'][0]),format='png',dpi=300,bbox_inches='tight')

#%% ============================================================
# LINEAR-NONLINEAR-POISSON (LNP) MODEL WITH ADDITIVE POPULATION INPUT
# ============================================================
# Extended LNP model for natural image responses:
#
#   r(t) = f( k · s(t)  +  γ · P(t)  +  b )
#
#   k · s(t)  : stimulus drive — linear projection of image t through
#               the spatial RF filter k (already estimated via Ridge regression)
#   γ · P(t)  : additive excitability boost from the population rate P(t),
#               computed as the mean z-scored activity of all OTHER neurons
#               (leave-one-out) to avoid circularity
#   b         : input bias — shifts the operating point of the nonlinearity
#   f(·)      : static output nonlinearity; fitted here as softplus and
#               power-law, with the best model selected by R²
#
# Stage 1 (done above): linear RF filter k estimated via Ridge regression.
#   ses.cRF[Ly, Lx, N] stores the filter for each neuron.
# Stage 2 (here):       compute g_stim = X_normalised @ k (stimulus drive),
#                       compute P(t) (population rate), then fit γ, b and
#                       the nonlinearity shape parameter by minimising MSE.
#                       The filter k is held FIXED — only 3–4 scalar params
#                       are free, making the optimisation fast.
#
# Diagnostic figure panels:
#   (1) Linear RF filter k — spatial structure of excitatory/suppressive zones
#   (2) Inferred input distributions — stimulus-only vs. total generator signal
#   (3) Fitted nonlinearity f(u) — empirical binned means + fitted curve
#   (4) Predicted vs. observed scatter — one dot per trial, R² annotated
#   (5) Input-space decomposition — stimulus drive vs. population drive,
#       coloured by observed response; tests whether axes are orthogonal
#   (6) Time-series overlay — first 200 trials, observed vs. predicted
# ============================================================

from scipy.optimize import minimize
from utils.nonlin_lib import *

# ---- (A) Select session and example neuron --------------------------------
# Pick the V1 neuron whose linear RF explains the most response variance.
# This is the neuron we expect to have the most interpretable LNP fit.
sesidx  = 0
ses     = sessions[sesidx]

# idx_V1  = np.where(ses.celldata['roi_name'] == 'V1')[0]
# iN      = idx_V1[np.argmax(ses.celldata['RF_R2'].values[idx_V1])]
idx_V1  = np.where(np.logical_and(ses.celldata['roi_name'] == 'V1',
                                  ses.celldata['RF_R2']>np.nanpercentile(ses.celldata['RF_R2'],90)))[0]
iN = np.random.choice(idx_V1)
# iN      = idx_V1[np.argmax(ses.celldata['RF_R2'].values[idx_V1])]
print(f'Example neuron — index: {iN} | cell_id: {ses.celldata["cell_id"][iN]}'
      f' | linear RF R² = {ses.celldata["RF_R2"][iN]:.3f}')

# ---- (B) Reconstruct the pixel design matrix X ----------------------------
# X must be built with the SAME subsampling factor (nsub) and pixel-column
# normalisation used inside linear_RF_cv so that ses.cRF @ X_row reproduces
# the linear prediction.
#   IMdata_ses  (H, W, K)  — images presented in this session
#   IMdata_sub  (Ly, Lx, K) — after spatial subsampling
#   X           (K, Ly·Lx) — row = one image, column = one pixel, normalised
IMdata_ses  = natimgdata[:, :, ses.trialdata['ImageNumber']]  # (H, W, K)
IMdata_sub  = IMdata_ses[::nsub, ::nsub, :]                   # downsample
Ly, Lx, K  = IMdata_sub.shape
X           = np.reshape(IMdata_sub, (Ly * Lx, K)).T          # (K, Ly·Lx)
X           = X / np.linalg.norm(X, axis=0)                   # pixel-column norm

# ---- (C) Z-score responses (same preprocessing as in the RF fitting loop) -
# zscore across trials (axis=0) so each neuron has zero mean and unit std.
from sklearn import preprocessing

# min_max_scaler = preprocessing.MinMaxScaler()
resp_all = preprocessing.minmax_scale(
    ses.respmat.T, feature_range=(0, 1), axis=0, copy=True)
resp_iN     = resp_all[:, iN]                 # (K,)  target neuron

# ---- (D) Stimulus generator signal ----------------------------------------
# g_stim(t) = X[t, :] · k  is the linear stimulus drive on trial t.
# Geometrically: how much does image t look like the neuron's preferred pattern?
k           = ses.cRF[:, :, iN].flatten()     # (Ly·Lx,) vectorised RF filter
g_stim      = X @ k                           # (K,)

# ---- (E) Population rate (leave-one-out) ----------------------------------
# P(t) = mean z-scored activity of every other neuron on trial t.
# Excluding neuron iN prevents the target signal from inflating γ.
# Standardising P to zero mean / unit std puts γ in the same units as g_stim.
idx_other   = np.arange(len(ses.celldata)) != iN
P           = np.mean(resp_all[:, idx_other], axis=1)   # (K,)
P           = zscore(P)                                  # standardise
# P = preprocessing.minmax_scale(
    # P, feature_range=(0, 1), axis=0, copy=True)

# ---- DIAGNOSIS: why the previous joint-MSE approach failed -----------------
# The Ridge RF k was fitted on z-scored responses → g_stim = X @ k lives in
# z-score units (~[-3, 3]).  resp_iN is min-max scaled to [0, 1].  Softplus
# and power-law are UNBOUNDED above, so for large positive u they output >> 1
# while the target is capped at 1.  Jointly optimising γ, b and the shape
# parameter over all K trials then gets stuck: the shape param must compress
# the output to [0, 1] but the flat gradient at large u makes this hard for
# L-BFGS-B starting from beta=5 (almost a ReLU).
# The sigmoid (bounded to [0, a]) is the only NL that naturally matches the
# [0,1] target — but it still fails when the generator-signal amplitude is
# wrong, because the transition region (slope > 0) falls outside the data.
#
# Fix — classical LNP estimation (no analytical NL derivatives needed):
#   Stage 1: OLS for γ and b  (exact, closed form)
#   Stage 2: compute u = g_stim + γP + b, bin it, compute E[resp | u] per bin
#   Stage 3: fit parametric NL shape to the 30 binned points via curve_fit
#            (Levenberg-Marquardt, ~1-2 params → always converges)
#
# This guarantees the fitted curve passes through what is SHOWN in the plot.
# ---- Print diagnostic scale information -----------------------------------
_X_diag = np.column_stack([g_stim, P, np.ones(K)])
_ls_diag = np.linalg.lstsq(_X_diag, resp_iN, rcond=None)[0]
print('=== SCALE DIAGNOSTICS ===')
print(f'  g_stim  range: [{g_stim.min():.3f}, {g_stim.max():.3f}]  '
      f'std={g_stim.std():.3f}  (z-score units from Ridge RF)')
print(f'  resp_iN range: [{resp_iN.min():.3f}, {resp_iN.max():.3f}]  '
      f'(min-max scaled to [0,1])')
print(f'  Linear warm-start:  scale_g={_ls_diag[0]:.4f}  γ={_ls_diag[1]:.4f}  '
      f'b={_ls_diag[2]:.4f}  R²={r2_score(resp_iN, _X_diag @ _ls_diag):.3f}')
print(f'  → g_stim must be rescaled by ×{_ls_diag[0]:.4f} to match resp_iN units')
print('=========================')

from scipy.optimize import curve_fit as _scipy_curve_fit

def _fit_nonlin_with_popcoupling(g_stim, P, resp, nl_func, p0_shape, bounds_shape,
                                  n_bins=30):
    """
    Robust three-stage nonlinearity fit — no analytical NL derivatives required.

    Stage 1 — OLS (exact):
        Regress resp on [g_stim, P, 1] to find γ and b in closed form.
        γ captures the linear population-coupling contribution; b is the bias.
        No optimisation needed — lstsq gives the global minimum.

    Stage 2 — Empirical nonlinearity:
        Compute u = g_stim + γ·P + b (total generator signal).
        Divide u into n_bins quantile bins and compute the conditional mean
        response E[resp | u] per bin.  These 30 points ARE what is plotted
        as the empirical nonlinearity in the diagnostic figure.

    Stage 3 — Parametric fit to binned means (curve_fit):
        Fit nl_func to the (bin_center, bin_mean) pairs using Levenberg-
        Marquardt.  With ~30 data points and 1–2 shape params this is
        overdetermined, fast, and globally convergent — avoiding the flat-
        landscape problem of the full-K MSE optimisation.

    Note: g_stim (z-score units) and resp ([0,1]) are on different scales.
    The bias b absorbs the constant offset; the shape param absorbs the gain.
    OLS finds the right b without the scale causing divergence.
    """
    K = len(resp)

    # Stage 1 — OLS for γ and b (closed form, no iteration)
    X_lin = np.column_stack([g_stim, P, np.ones(K)])
    ls_coef, _, _, _ = np.linalg.lstsq(X_lin, resp, rcond=None)
    gamma_fit = float(ls_coef[1])
    b_fit     = float(ls_coef[2])

    # Stage 2 — Total generator signal and empirical conditional means
    u         = g_stim + gamma_fit * P + b_fit
    bin_edges = np.percentile(u, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)            # remove duplicate percentile edges
    n_b       = len(bin_edges) - 1
    bin_ctr   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_mean  = np.array([
        resp[(u >= bin_edges[i]) & (u < bin_edges[i + 1])].mean()
        for i in range(n_b)])

    # Stage 3 — Fit parametric NL shape to binned means via curve_fit
    # Analytical NL derivatives are NOT needed here — LM uses numerical Jacobian.
    if p0_shape:
        blo = [b[0] if b[0] is not None else -np.inf for b in bounds_shape]
        bhi = [b[1] if b[1] is not None else  np.inf for b in bounds_shape]
        try:
            popt, _ = _scipy_curve_fit(
                nl_func, bin_ctr, bin_mean,
                p0=p0_shape, bounds=(blo, bhi), maxfev=10000)
            shape_fit = list(popt)
        except (RuntimeError, ValueError):
            shape_fit = list(p0_shape)      # fall back to p0 if curve_fit fails
    else:
        shape_fit = []

    pred = nl_func(u, *shape_fit) if shape_fit else nl_func(u)
    r2   = float(r2_score(resp, pred))

    return dict(gamma=gamma_fit, b=b_fit, shape_p=shape_fit,
                pred=pred, u=u, r2=r2,
                bin_ctr=bin_ctr, bin_mean=bin_mean)

# Candidate nonlinearities:  (name, function, p0_shape, bounds_shape)
# Softplus: f(u) = (1/β) log(1 + exp(β·u)) — smooth threshold nonlinearity
# Power-law: f(u) = max(0, u)^p — supralinear for p>1
nl_candidates = [
    ('Softplus',  nl_softplus, [5.0], [(0.01, 50.0)]),
    ('Power-law', nl_powerlaw, [2.0], [(0.1,  4.0)]),
    # ('Exp', nl_exp, [0.0], [(None,  None)]),
    ('Sigmoid',   nl_sigmoid,   [1.0], [(0,  None)]),
]

nl_results = {}
for nl_name, nl_func, p0_s, bnds_s in nl_candidates:
    nl_results[nl_name] = _fit_nonlin_with_popcoupling(
        g_stim, P, resp_iN, nl_func, p0_s, bnds_s)
    r = nl_results[nl_name]
    print(f'  {nl_name:12s}  R²={r["r2"]:.3f}  γ={r["gamma"]:.3f}  '
          f'b={r["b"]:.3f}  shape={[round(v, 3) for v in r["shape_p"]]}')

# Best nonlinearity by R²
best_name   = max(nl_results, key=lambda n: nl_results[n]['r2'])
res         = nl_results[best_name]
nl_func_b   = {n: f for n, f, _, _ in nl_candidates}[best_name]
print(f'\nBest model: {best_name}  |  R² = {res["r2"]:.3f}')

# ---- (G) Diagnostic figure ------------------------------------------------
fig = plt.figure(figsize=(15, 8))
fig.suptitle(
    f'LNP Diagnostic  |  Session: {ses.session_id}'
    f'  |  Neuron: {ses.celldata["cell_id"][iN]}  (V1)\n'
    f'Linear RF R² = {ses.celldata["RF_R2"][iN]:.3f}   |   '
    f'LNP R² = {res["r2"]:.3f}   |   '
    f'Model: {best_name}   |   γ = {res["gamma"]:.3f}   |   b = {res["b"]:.3f}',
    fontsize=9)

# Panel 1 — Linear RF filter -------------------------------------------------
# Positive weights (red): stimulus features that excite the neuron.
# Negative weights (blue): suppressive surroundings.
# The spatial structure here is the same filter used to compute g_stim(t).
ax1 = fig.add_subplot(2, 3, 1)
lim = np.max(np.abs(ses.cRF[:, :, iN])) * 1.2
ax1.imshow(ses.cRF[:, :, iN], cmap='bwr', vmin=-lim, vmax=lim, aspect='auto')
ax1.set_title('(1) Linear RF filter  k', fontsize=10)
ax1.set_xlabel(f'Azimuth  (subsampled ×{nsub})')
ax1.set_ylabel(f'Elevation  (subsampled ×{nsub})')
ax1.set_xticks([]); ax1.set_yticks([])

# Panel 2 — Inferred input distributions ------------------------------------
# Blue: stimulus-only drive k·s(t), showing the range of visual inputs.
# Orange: total generator signal u = k·s + γ·P + b — shifted by the mean
#   population rate contribution.  The spread of u relative to the threshold
#   determines how much of the nonlinearity the neuron explores.
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(g_stim,   bins=60, alpha=0.6, color='royalblue',
         label=r'Stimulus drive  $k \cdot s(t)$')
ax2.hist(res['u'], bins=60, alpha=0.6, color='darkorange',
         label=r'Total input  $k \cdot s + \gamma P + b$')
ax2.axvline(0, color='k', lw=0.8, ls='--', label='u = 0 (threshold)')
ax2.set_xlabel('Generator signal  u')
ax2.set_ylabel('Trial count')
ax2.set_title('(2) Inferred inputs across all images', fontsize=10)
ax2.legend(fontsize=7, frameon=False)

# Panel 3 — Fitted nonlinearity f(u) ----------------------------------------
# Black dots: empirical mean response binned by the total generator signal.
#   These are the "ground truth" input–output data points.
# Red curve: the analytical nonlinearity f(u) evaluated over the observed
#   input range.  Good fit = curve passes through the dots.
ax3 = fig.add_subplot(2, 3, 3)
# Use the binned means computed INSIDE _fit_nonlin_with_popcoupling —
# the parametric curve was fitted to exactly these points, so they are
# guaranteed to match (no re-binning with a different n_bins).
bin_centers = res['bin_ctr']
bin_means   = res['bin_mean']
ax3.scatter(bin_centers, bin_means, s=35, color='k', zorder=3,
            label='Empirical (binned mean)')
u_range  = np.linspace(res['u'].min(), res['u'].max(), 300)
nl_curve = nl_func_b(u_range, *res['shape_p']) if res['shape_p'] else nl_func_b(u_range)
ax3.plot(u_range, nl_curve, 'r-', lw=2, label=f'Fitted {best_name}')
ax3.axvline(0, color='gray', lw=0.8, ls='--')
ax3.set_xlabel('Generator signal  u')
ax3.set_ylabel('Response')
ax3.set_title(f'(3) Nonlinearity  f(u) — {best_name}', fontsize=10)
ax3.legend(fontsize=7, frameon=False)

# Panel 4 — Predicted vs. observed scatter -----------------------------------
# Each point is one trial.  A 1:1 diagonal (dashed red) marks perfect
# prediction.  Systematic deviations indicate the nonlinearity is mis-specified
# or that additional sources of variability (e.g., noise correlations) are
# not captured.
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(res['pred'], resp_iN, s=2, alpha=0.25, color='steelblue')
p_lo = min(res['pred'].min(), resp_iN.min())
p_hi = max(res['pred'].max(), resp_iN.max())
ax4.plot([p_lo, p_hi], [p_lo, p_hi], 'r--', lw=1.2)
ax4.text(0.05, 0.92, f'R² = {res["r2"]:.3f}', transform=ax4.transAxes, fontsize=10,
         color='k', fontweight='bold')
ax4.set_xlabel('Predicted response')
ax4.set_ylabel('Observed response (z-scored)')
ax4.set_title('(4) Predicted vs. Observed', fontsize=10)

# Panel 5 — Input-space decomposition ----------------------------------------
# x-axis: stimulus drive k·s(t) — determined by what image was shown.
# y-axis: population drive γ·P(t) — determined by the network's internal state.
# Colour: observed response.  If the two axes are orthogonal (no correlation)
# and γ > 0, the response surface follows contours of constant u = x + y,
# i.e. diagonal iso-response lines — the hallmark of additive input modulation.
ax5 = fig.add_subplot(2, 3, 5)
gamma_P = res['gamma'] * P          # scaled population rate contribution
sc = ax5.scatter(g_stim, gamma_P, s=2, alpha=0.3, c=resp_iN,
                 cmap='RdBu_r',
                 vmin=np.percentile(resp_iN, 5),
                 vmax=np.percentile(resp_iN, 95))
plt.colorbar(sc, ax=ax5, label='Observed response (z-scored)', pad=0.02)
ax5.axhline(0, color='k', lw=0.5, ls='--')
ax5.axvline(0, color='k', lw=0.5, ls='--')
ax5.set_xlabel(r'Stimulus drive  $k \cdot s(t)$')
ax5.set_ylabel(rf'Population drive  $\gamma \cdot P(t)$  ($\gamma$ = {res["gamma"]:.2f})')
ax5.set_title('(5) Input-space decomposition\n(colour = observed response)', fontsize=10)

# Panel 6 — Time-series overlay (first N_show trials) -----------------------
# Observed (black) and predicted (red) responses aligned by trial index.
# Good fits track both the slow trends (population-rate fluctuations) and
# the fast image-specific peaks.  Residuals reflect private noise.
ax6 = fig.add_subplot(2, 3, 6)
N_show = min(200, K)
t_ax   = np.arange(N_show)
ax6.plot(t_ax, resp_iN[:N_show], color='k', lw=0.8, alpha=0.85, label='Observed')
ax6.plot(t_ax, res['pred'][:N_show], color='r', lw=0.8, alpha=0.85,
         label=f'Predicted ({best_name})')
ax6.set_xlabel('Trial #')
ax6.set_ylabel('Response (z-scored)')
ax6.set_title(f'(6) Time-series — first {N_show} trials', fontsize=10)
ax6.legend(fontsize=7, frameon=False)
ax6.set_xlim([0, N_show])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3)
plt.savefig(os.path.join(figdir, f'LNP_diagnostic_{ses.session_id}.png'),
            dpi=150, bbox_inches='tight')
plt.show()


#%% ============================================================
# FULL LNP FIT: JOINT OPTIMISATION OF RF FILTER k + NONLINEARITY (± POP. COUPLING)
# ============================================================
# Model:
#   r(t) = f(  X[t,:] @ k  +  γ·P(t)  +  b  )     if fit_popcoupling=True
#   r(t) = f(  X[t,:] @ k              +  b  )     if fit_popcoupling=False
#
# Three bugs in the previous attempt caused negative R²:
#
#  Bug 1 — No analytical gradient (MAIN CAUSE)
#    scipy L-BFGS-B without jac= uses finite differences: one extra forward
#    pass per parameter.  With n_pix ~ 3000–16000, each gradient step costs
#    n_pix × O(K·n_pix) operations → extremely slow and numerically inaccurate.
#    The optimizer effectively stalls after a few steps.
#    Fix: provide jac= with an analytic gradient for k, γ, b.  Shape params
#    (only 1–2 scalars) use cheap central-difference numerical gradient.
#
#  Bug 2 — k warm-start not rescaled (AMPLIFIES Bug 1)
#    The warm-start regression finds ls_coef[0] = optimal linear scale of
#    X @ k_init to predict resp, but k_init itself was NOT scaled by ls_coef[0].
#    The generator signal X @ k_init was therefore in the wrong amplitude range,
#    so the optimizer had to both find the right scale for k AND optimise the
#    nonlinearity simultaneously — a badly conditioned starting point.
#    Fix: k_init = k_seed * ls_coef[0] before passing to the optimizer.
#
#  Bug 3 — Ridge interacts with scale error (CAUSED COLLAPSE)
#    When k must grow 10× to reach the right amplitude, ||k||² grows 100×
#    and the ridge penalty dominates → optimizer drives k back toward zero
#    → near-constant predictions → R² < 0.
#    Fix: scale correction (Bug 2 fix) prevents k from growing excessively.
#
# Analytic gradient derivation:
#   L = (1/K) Σ (resp - f(u))²  +  λ ||k||²   where  u = X@k + γ·P + b
#   δ = (pred - resp) * f'(u)                  (K,) error weighted by NL slope
#   ∂L/∂k     = (2/K) X.T @ δ  +  2λ k        (n_pix,) — one matmul, O(K·n_pix)
#   ∂L/∂γ     = (2/K) P.T @ δ                  scalar
#   ∂L/∂b     = (2/K) sum(δ)                   scalar
#   ∂L/∂shape ≈ central differences            1–2 scalars, cheap
# ============================================================

# idx_V1  = np.where(np.logical_and(ses.celldata['roi_name'] == 'V1',
                                #   ses.celldata['RF_R2']>np.nanpercentile(ses.celldata['RF_R2'],90)))[0]
# iN = np.random.choice(idx_V1)

# ---- (B) Reconstruct the pixel design matrix X ----------------------------
# X must be built with the SAME subsampling factor (nsub) and pixel-column
# normalisation used inside linear_RF_cv so that ses.cRF @ X_row reproduces
# the linear prediction.
#   IMdata_ses  (H, W, K)  — images presented in this session
#   IMdata_sub  (Ly, Lx, K) — after spatial subsampling
#   X           (K, Ly·Lx) — row = one image, column = one pixel, normalised
IMdata_ses  = natimgdata[:, :, ses.trialdata['ImageNumber']]  # (H, W, K)
IMdata_sub  = IMdata_ses[::nsub, ::nsub, :]                   # downsample
Ly, Lx, K  = IMdata_sub.shape
X           = np.reshape(IMdata_sub, (Ly * Lx, K)).T          # (K, Ly·Lx)
X           = X / np.linalg.norm(X, axis=0)                   # pixel-column norm

# ---- (C) Z-score responses (same preprocessing as in the RF fitting loop) -
# zscore across trials (axis=0) so each neuron has zero mean and unit std.
from sklearn import preprocessing

# min_max_scaler = preprocessing.MinMaxScaler()
resp_all = preprocessing.minmax_scale(
    ses.respmat.T, feature_range=(0, 1), axis=0, copy=True)
resp_iN     = resp_all[:, iN]                 # (K,)  target neuron

# # ---- (D) Stimulus generator signal ----------------------------------------
# # g_stim(t) = X[t, :] · k  is the linear stimulus drive on trial t.
# # Geometrically: how much does image t look like the neuron's preferred pattern?
# k           = ses.cRF[:, :, iN].flatten()     # (Ly·Lx,) vectorised RF filter
# g_stim      = X @ k                           # (K,)

# ---- (E) Population rate (leave-one-out) ----------------------------------
# P(t) = mean z-scored activity of every other neuron on trial t.
# Excluding neuron iN prevents the target signal from inflating γ.
# Standardising P to zero mean / unit std puts γ in the same units as g_stim.
idx_other   = np.arange(len(ses.celldata)) != iN
P           = np.mean(resp_all[:, idx_other], axis=1)   # (K,)
P           = zscore(P)                                  # standardise
# P = preprocessing.minmax_scale(
    # P, feature_range=(0, 1), axis=0, copy=True)

# ---- USER-FACING FLAGS -------------------------------------------------------
fit_popcoupling = True    # True  → fit γ·P(t) term;  False → pure stimulus LNP
lam_ridge       = 0.0001   # L2 regularisation on k (larger = smoother RF)
# ------------------------------------------------------------------------------
# NOTE: ses, iN, X, P, resp_iN, K, Ly, Lx, nl_candidates, k_seed (Ridge RF),
#       g_stim (from the fixed-k section above) are all reused here.
# ------------------------------------------------------------------------------

# ---- Derivative functions for each nonlinearity (needed for analytic grad) --
def _dnl_softplus(u, beta):
    """∂f/∂u for softplus f(u,β) = (1/β) log(1+exp(βu)):  σ(βu)"""
    b = np.abs(beta) + 1e-4
    return 1.0 / (1.0 + np.exp(-b * np.clip(u, -100.0, 100.0)))

def _dnl_powerlaw(u, p):
    """∂f/∂u for power-law f(u,p) = max(0,u)^|p|:  |p|·max(0,u)^(|p|-1)"""
    p_abs = np.abs(p) + 1e-4
    return p_abs * np.power(np.maximum(u, 0.0) + 1e-12, p_abs - 1.0) * (u > 0)

def _dnl_sigmoid(u, a):
    hardcodeslope = 5
    """∂f/∂u for sigmoid f(u,a) = |a|·σ(5u):  5·|a|·σ(5u)·(1 - σ(5u))
    The factor 5 matches the hardcoded slope in nl_sigmoid (nonlin_lib.py)."""
    sig = 1.0 / (1.0 + np.exp(-hardcodeslope * np.clip(u, -100.0, 100.0)))
    return hardcodeslope * np.abs(a) * sig * (1.0 - sig)

# Map nl names → derivative functions (extend if more NL types are added)
_dnl_map = {'Softplus': _dnl_softplus, 'Power-law': _dnl_powerlaw,
            'Sigmoid': _dnl_sigmoid}

def _make_loss_and_grad(X, P, resp, nl_func, dnl_func,
                        n_pix, n_shape, fit_popcoupling, lam_ridge):
    """
    Factory returning (loss, gradient) for L-BFGS-B's jac= argument.

    Gradient for k, γ, b is analytic (cost = one extra X.T @ δ matmul).
    Gradient for shape params is numerical (central differences, 1–2 evals).

    Parameter vector layout (fit_popcoupling=True):
        params = [ k (n_pix)  |  gamma (1)  |  b (1)  |  shape_p (n_shape) ]
    Parameter vector layout (fit_popcoupling=False):
        params = [ k (n_pix)  |  b (1)  |  shape_p (n_shape) ]
    """
    K       = len(resp)
    eps_sh  = 1e-5   # step size for numerical shape-param gradient

    def _f_and_g(params):
        k_p = params[:n_pix]
        if fit_popcoupling:
            gamma_p = params[n_pix]
            b_p     = params[n_pix + 1]
            sh_p    = list(params[n_pix + 2: n_pix + 2 + n_shape])
        else:
            gamma_p = 0.0
            b_p     = params[n_pix]
            sh_p    = list(params[n_pix + 1: n_pix + 1 + n_shape])

        u    = X @ k_p + gamma_p * P + b_p
        pred = nl_func(u, *sh_p) if n_shape else nl_func(u)
        nl_d = dnl_func(u, *sh_p) if n_shape else dnl_func(u)

        resid = resp - pred
        mse   = float(np.mean(resid ** 2))
        reg   = lam_ridge * float(np.dot(k_p, k_p))
        loss  = mse + reg

        # Analytic gradient: δ = (pred - resp) * f'(u)  →  scale = 2/K
        delta  = (2.0 / K) * (-resid) * nl_d             # (K,)
        grad_k = X.T @ delta + 2.0 * lam_ridge * k_p     # (n_pix,)

        if fit_popcoupling:
            grad_gamma = float(np.dot(P, delta))
            grad_b     = float(np.sum(delta))
        else:
            grad_b = float(np.sum(delta))

        # Numerical gradient for shape params (only 1–2 scalars → cheap)
        grad_sh = np.zeros(n_shape)
        for j in range(n_shape):
            sh_hi = sh_p[:]; sh_hi[j] += eps_sh
            sh_lo = sh_p[:]; sh_lo[j] -= eps_sh
            pred_hi = nl_func(u, *sh_hi) if n_shape else nl_func(u)
            pred_lo = nl_func(u, *sh_lo) if n_shape else nl_func(u)
            grad_sh[j] = (float(np.mean((resp - pred_hi) ** 2))
                        - float(np.mean((resp - pred_lo) ** 2))) / (2.0 * eps_sh)

        if fit_popcoupling:
            grad = np.concatenate([grad_k, [grad_gamma, grad_b], grad_sh])
        else:
            grad = np.concatenate([grad_k, [grad_b], grad_sh])

        return loss, grad

    return _f_and_g

def fit_lnp_full(X, P, resp, nl_candidates, dnl_map,
                 fit_popcoupling=True, k_init=None, lam_ridge=0.01):
    """
    Jointly fit RF filter k, nonlinearity shape, bias b, and (optionally)
    population coupling γ using analytic gradients and a scale-corrected warm start.

    Parameters
    ----------
    X               : (K, n_pix)  pixel design matrix (same normalisation as linear_RF_cv)
    P               : (K,)        population rate (z-scored, leave-one-out)
    resp            : (K,)        z-scored neural responses
    nl_candidates   : list of (name, nl_func, p0_shape, bounds_shape)
    dnl_map         : dict  {name: dnl_func}  derivative of nl_func w.r.t. u
    fit_popcoupling : bool        if True, γ is free; else γ ≡ 0
    k_init          : (n_pix,) or None  seed RF; None → STA
    lam_ridge       : float       L2 penalty on k

    Returns
    -------
    best_name : str
    all_results : dict {name: dict(k, gamma, b, shape_p, u, pred, r2)}
    """
    n_pix = X.shape[1]
    K     = len(resp)

    # ---- Seed k from STA if not provided ------------------------------------
    if k_init is None:
        w      = resp - resp.min() + 1e-6
        k_init = (X.T @ w) / w.sum()   # STA weighted by non-negative response

    # ---- Scale-corrected warm start -----------------------------------------
    # Regress resp on [g_init, P, 1] to find the optimal linear scale of g_init.
    # Multiply k_init by this scale so the generator signal starts in the
    # correct amplitude range — without this, Bug 2 causes the optimizer to
    # waste iterations rescaling k while the ridge penalty fights back.
    g_init = X @ k_init
    if fit_popcoupling:
        X_lin = np.column_stack([g_init, P, np.ones(K)])
    else:
        X_lin = np.column_stack([g_init, np.ones(K)])
    ls_coef, _, _, _ = np.linalg.lstsq(X_lin, resp, rcond=None)

    scale_k = float(ls_coef[0])
    k_init  = k_init * scale_k          # apply scale so X @ k_init is in resp range
    gamma0  = float(ls_coef[1]) if fit_popcoupling else 0.0
    b0      = float(ls_coef[1 + int(fit_popcoupling)])

    print(f'  [warm start] k scale correction: ×{scale_k:.4f} | '
          f'linear R² = {r2_score(resp, X_lin @ ls_coef):.3f}')

    best_r2, best_name = -np.inf, None
    all_results = {}

    for nl_name, nl_func, p0_shape, bounds_shape in nl_candidates:
        n_shape  = len(p0_shape)
        dnl_func = dnl_map.get(nl_name)
        if dnl_func is None:
            print(f'  Warning: no derivative defined for {nl_name}, skipping.')
            continue

        if fit_popcoupling:
            p0     = np.concatenate([k_init, [gamma0, b0], p0_shape])
            bounds = ([(None, None)] * n_pix
                    + [(None, None)]           # gamma
                    + [(None, None)]           # b
                    + list(bounds_shape))
        else:
            p0     = np.concatenate([k_init, [b0], p0_shape])
            bounds = ([(None, None)] * n_pix
                    + [(None, None)]           # b
                    + list(bounds_shape))

        f_and_g = _make_loss_and_grad(X, P, resp, nl_func, dnl_func,
                                      n_pix, n_shape, fit_popcoupling, lam_ridge)
        try:
            opt = minimize(f_and_g, p0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})

            k_fit = opt.x[:n_pix]
            if fit_popcoupling:
                gamma_fit = float(opt.x[n_pix])
                b_fit     = float(opt.x[n_pix + 1])
                shape_fit = list(opt.x[n_pix + 2: n_pix + 2 + n_shape])
            else:
                gamma_fit = 0.0
                b_fit     = float(opt.x[n_pix])
                shape_fit = list(opt.x[n_pix + 1: n_pix + 1 + n_shape])

            u    = X @ k_fit + gamma_fit * P + b_fit
            pred = nl_func(u, *shape_fit) if shape_fit else nl_func(u)
            r2   = float(r2_score(resp, pred))

            all_results[nl_name] = dict(k=k_fit, gamma=gamma_fit, b=b_fit,
                                        shape_p=shape_fit, u=u, pred=pred, r2=r2,
                                        nit=opt.nit, nfev=opt.nfev,
                                        converged=opt.success)
            if r2 > best_r2:
                best_r2, best_name = r2, nl_name
            print(f'  {nl_name:12s}  R²={r2:.3f}  γ={gamma_fit:.3f}  b={b_fit:.3f}  '
                  f'shape={[round(v,3) for v in shape_fit]}  '
                  f'iters={opt.nit}  feval={opt.nfev}  ok={opt.success}')

        except Exception as e:
            print(f'  Warning: {nl_name} failed: {e}')
            all_results[nl_name] = dict(k=None, gamma=np.nan, b=np.nan,
                                        shape_p=[], u=None, pred=None, r2=np.nan,
                                        nit=0, nfev=0, converged=False)

    return best_name, all_results

# ---- Run the full joint fit --------------------------------------------------
k_seed = ses.cRF[:, :, iN].flatten()   # Ridge RF as seed (optimiser is free to move)

coupling_label = 'with population coupling  (γ free)' if fit_popcoupling \
                 else 'without population coupling  (γ = 0)'
print(f'\nFitting full LNP  [{coupling_label},  λ={lam_ridge}]'
      f'  |  n_pix={X.shape[1]}  K={K}')

best_name_full, nl_results_full = fit_lnp_full(
    X, P, resp_iN,
    nl_candidates   = nl_candidates,
    dnl_map         = _dnl_map,
    fit_popcoupling = fit_popcoupling,
    k_init          = k_seed,
    # k_init          = None,
    lam_ridge       = lam_ridge)

res_full    = nl_results_full[best_name_full]
nl_func_full = {n: f for n, f, _, _ in nl_candidates}[best_name_full]
k_2d        = res_full['k'].reshape(Ly, Lx)   # reshape for 2-D display
g_stim_full = X @ res_full['k']               # stimulus drive with the new k

print(f'\nBest full-LNP model: {best_name_full}  |  R² = {res_full["r2"]:.3f}')

# ---- Diagnostic figure -------------------------------------------------------
fig = plt.figure(figsize=(15, 8))
fig.suptitle(
    f'Full LNP Diagnostic  [{coupling_label}]\n'
    f'Session: {ses.session_id}  |  Neuron: {ses.celldata["cell_id"][iN]}  (V1)  |  '
    f'R² = {res_full["r2"]:.3f}  |  Model: {best_name_full}  |  '
    f'γ = {res_full["gamma"]:.3f}  |  b = {res_full["b"]:.3f}  |  λ = {lam_ridge}',
    fontsize=9)

# Panel 1 — Jointly fitted RF filter -----------------------------------------
# k is no longer the linear-Ridge RF — it is the filter that minimises the
# nonlinear loss.  Comparing this to the Ridge RF (panel 1 of the fixed-k
# figure) reveals what the nonlinearity "reshapes" the effective RF into.
ax1 = fig.add_subplot(2, 3, 1)
lim = np.max(np.abs(k_2d)) * 1.2
ax1.imshow(k_2d, cmap='bwr', vmin=-lim, vmax=lim, aspect='auto')
ax1.set_title('(1) Jointly fitted RF filter  k', fontsize=10)
ax1.set_xlabel(f'Azimuth  (subsampled ×{nsub})')
ax1.set_ylabel(f'Elevation  (subsampled ×{nsub})')
ax1.set_xticks([]); ax1.set_yticks([])

# Panel 2 — Inferred input distributions -------------------------------------
# Blue: stimulus drive from the jointly fitted k.  Orange: total generator
# signal u = k·s + γP + b.  Compare the spread and shift to the fixed-k
# version to see what joint optimisation changed.
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(g_stim_full,   bins=60, alpha=0.6, color='royalblue',
         label=r'Stimulus drive  $k \cdot s(t)$')
ax2.hist(res_full['u'], bins=60, alpha=0.6, color='darkorange',
         label=(r'Total input  $k \cdot s + \gamma P + b$'
                if fit_popcoupling else r'Total input  $k \cdot s + b$'))
ax2.axvline(0, color='k', lw=0.8, ls='--')
ax2.set_xlabel('Generator signal  u')
ax2.set_ylabel('Trial count')
ax2.set_title('(2) Inferred inputs across all images', fontsize=10)
ax2.legend(fontsize=7, frameon=False)

# Panel 3 — Fitted nonlinearity f(u) -----------------------------------------
# Black dots: empirical mean response binned by the total generator signal.
# Red curve: analytical nonlinearity.  Because k and f are fitted jointly,
# the generator signal is stretched/compressed so that f can best explain
# the data — the resulting f(u) may look different from the fixed-k version.
ax3 = fig.add_subplot(2, 3, 3)
n_bins      = 25
bin_edges   = np.percentile(res_full['u'], np.linspace(0, 100, n_bins + 1))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_means   = np.array([
    resp_iN[(res_full['u'] >= bin_edges[i]) & (res_full['u'] < bin_edges[i + 1])].mean()
    for i in range(n_bins)])
ax3.scatter(bin_centers, bin_means, s=35, color='k', zorder=3,
            label='Empirical (binned mean)')
u_range  = np.linspace(res_full['u'].min(), res_full['u'].max(), 300)
nl_curve = (nl_func_full(u_range, *res_full['shape_p'])
            if res_full['shape_p'] else nl_func_full(u_range))
ax3.plot(u_range, nl_curve, 'r-', lw=2, label=f'Fitted {best_name_full}')
ax3.axvline(0, color='gray', lw=0.8, ls='--')
ax3.set_xlabel('Generator signal  u')
ax3.set_ylabel('Response (z-scored)')
ax3.set_title(f'(3) Nonlinearity  f(u) — {best_name_full}', fontsize=10)
ax3.legend(fontsize=7, frameon=False)

# Panel 4 — Predicted vs. observed scatter ------------------------------------
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(res_full['pred'], resp_iN, s=2, alpha=0.25, color='steelblue')
p_lo = min(res_full['pred'].min(), resp_iN.min())
p_hi = max(res_full['pred'].max(), resp_iN.max())
ax4.plot([p_lo, p_hi], [p_lo, p_hi], 'r--', lw=1.2)
ax4.text(0.05, 0.92, f'R² = {res_full["r2"]:.3f}',
         transform=ax4.transAxes, fontsize=10, fontweight='bold')
ax4.set_xlabel('Predicted response')
ax4.set_ylabel('Observed response (z-scored)')
ax4.set_title('(4) Predicted vs. Observed', fontsize=10)

# Panel 5 — Input-space decomposition OR RF comparison (depends on flag) ------
ax5 = fig.add_subplot(2, 3, 5)
if fit_popcoupling:
    # Show stimulus drive (x) vs population drive (y), coloured by response.
    # Diagonal iso-response contours → additive input modulation.
    gamma_P_full = res_full['gamma'] * P
    sc = ax5.scatter(g_stim_full, gamma_P_full, s=2, alpha=0.3, c=resp_iN,
                     cmap='RdBu_r',
                     vmin=np.percentile(resp_iN, 5),
                     vmax=np.percentile(resp_iN, 95))
    plt.colorbar(sc, ax=ax5, label='Observed response (z-scored)', pad=0.02)
    ax5.axhline(0, color='k', lw=0.5, ls='--')
    ax5.axvline(0, color='k', lw=0.5, ls='--')
    ax5.set_xlabel(r'Stimulus drive  $k \cdot s(t)$')
    ax5.set_ylabel(rf'Population drive  $\gamma \cdot P(t)$'
                   rf'  ($\gamma$ = {res_full["gamma"]:.2f})')
    ax5.set_title('(5) Input-space decomposition\n(colour = observed response)',
                  fontsize=10)
else:
    # Compare the linear-RF-derived drive (previous section) with the
    # jointly fitted drive.  High correlation means the joint fit found a
    # similar RF; divergence indicates the nonlinearity reshaped the RF.
    ax5.scatter(g_stim, g_stim_full, s=2, alpha=0.3, color='k')
    r_kk = float(np.corrcoef(g_stim, g_stim_full)[0, 1])
    ax5.set_xlabel(r'Linear-RF drive  $k_{\rm Ridge} \cdot s(t)$')
    ax5.set_ylabel(r'Joint-LNP drive  $k_{\rm LNP} \cdot s(t)$')
    ax5.set_title('(5) RF drive comparison\n(Ridge vs. joint LNP)', fontsize=10)
    ax5.text(0.05, 0.92, f'r = {r_kk:.3f}', transform=ax5.transAxes, fontsize=10)

# Panel 6 — Time-series overlay -----------------------------------------------
ax6 = fig.add_subplot(2, 3, 6)
N_show = min(200, K)
t_ax   = np.arange(N_show)
ax6.plot(t_ax, resp_iN[:N_show], color='k', lw=0.8, alpha=0.85, label='Observed')
ax6.plot(t_ax, res_full['pred'][:N_show], color='r', lw=0.8, alpha=0.85,
         label=f'Predicted ({best_name_full}, full LNP)')
ax6.set_xlabel('Trial #')
ax6.set_ylabel('Response (z-scored)')
ax6.set_title(f'(6) Time-series — first {N_show} trials', fontsize=10)
ax6.legend(fontsize=7, frameon=False)
ax6.set_xlim([0, N_show])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3)
fname = (f'LNP_full_{"withpc" if fit_popcoupling else "nopc"}_{ses.session_id}.png')
plt.savefig(os.path.join(figdir, fname), dpi=150, bbox_inches='tight')
plt.show()
