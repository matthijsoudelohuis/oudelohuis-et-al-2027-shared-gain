#%% 
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import zscore,linregress

from utils.gain_lib import comp_poprate
from utils.plot_lib import *

#%% ###########################################################################
# NONLINEAR TRANSFER FUNCTION FITTING PIPELINE
# Model: r(t) = f( θ_k(t) + γ · P(t) + b )
#   θ_k  : stimulus drive — one free parameter per orientation (16)
#   γ    : population-rate scaling (additive input, 1 param)
#   b    : input bias (1 param)
#   f(·) : nonlinearity (with model-specific free parameters)
###########################################################################

#%% Define nonlinearities with fittable parameters
def nl_linear(u):
    return u

def nl_relu(u):
    return np.maximum(0.0, u)

def nl_softplus(u, beta=5):
    # f(u) = (1/β) log(1 + exp(β·u)); β controls sharpness (→ReLU as β→∞)
    b = np.abs(beta) + 1e-4
    bu = b * u
    return np.where(bu > 30.0, u, np.log1p(np.exp(np.clip(bu, -500.0, 30.0))) / b)

def nl_sigmoid(u, a=1):
    # maps sigmoid to [0, a]: f(u) = a · σ(u)
    return np.abs(a) / (1.0 + np.exp(5*-np.clip(u, -100.0, 100.0)))

def nl_tanh(u, a=1):
    # maps tanh's [-1,1] to [0, a]: f(u) = a · (1 + tanh(u)) / 2
    return np.abs(a) * 0.5 * (1.0 + np.tanh(u))

def nl_powerlaw(u, p=2):
    # f(u) = max(0,u)^p; p is the free exponent
    return np.power(np.maximum(0.0, u), np.abs(p) + 1e-4)

def nl_exp(u):
    # max(0, exp(u)-1), shifted so f(0)=0; output gain a is universal
    return np.maximum(0.0, np.expm1(np.clip(u, -500.0, 10.0)))

# Format: (name, nl_func, n_shape, p0_shape, bounds_shape)
# Responses are min-max normalised to [0,1] before fitting, so all nonlinearities
# operate in the same output regime without per-model gain/offset parameters.
NL_CONFIGS = [
    ('Linear',          nl_linear,   0, [],      []),
    ('ReLU',            nl_relu,     0, [],      []),
    ('Softplus',        nl_softplus, 1, [5.0],   [(0.01, 50.0)]),
    # ('Tanh',            nl_tanh,     1, [1.0],   [(0.0, None)]),
    ('Exp',             nl_exp,      0, [],      []),
    ('Power-law (p=2)', nl_powerlaw, 1, [2.0],   [(0.1,  4.0)]),
    ('Sigmoid',         nl_sigmoid,  1, [1],   [(0.0, None)]),
]

nl_names = [c[0] for c in NL_CONFIGS]
nNL      = len(NL_CONFIGS)
clrs_nl  = sns.color_palette('tab10', nNL)

#%%
def tuning_input(stim, pref=0.0, width=1.0, gain=1.0):
    return gain * np.exp(-(stim - pref)**2 / (2 * width**2))


#%%
def simulate_responses(
    x,
    nonlinearity,
    noise_std=0.2,
    n_trials=1000
):
    responses = []
    for _ in range(n_trials):
        noisy_x = x + np.random.normal(0, noise_std, size=x.shape)
        y = nonlinearity(noisy_x)
        responses.append(y)
    return np.array(responses)

#%% Core fitting function

def fit_nl_models(resp, stim_ids, poprate, configs=NL_CONFIGS):
    """
    Fit all NL models to a single neuron's trial-by-trial responses.

    Model: r_norm = f( θ_k + γ·P + b )
      Responses are min-max normalised to [0,1] before fitting so all
      nonlinearities share the same output regime without per-model gain.
      Shared params: θ_k (nstim), γ, b  — warm-started via least squares.
      Per-model params: shape params only (e.g. softplus β, power-law p).

    Returns dict keyed by model name:
      r2, theta, gamma, b, nl_par, pred (in [0,1] space), u, resp_norm
    """
    nstim = int(stim_ids.max()) + 1
    nT    = len(resp)

    # Least-squares warm start on normalised responses
    X = np.zeros((nT, nstim + 2))
    for k in range(nstim):
        X[stim_ids == k, k] = 1.0
    X[:, nstim]     = poprate
    X[:, nstim + 1] = 1.0
    p_ls, _, _, _ = np.linalg.lstsq(X, resp, rcond=None)
    theta0 = p_ls[:nstim]
    gamma0 = p_ls[nstim]
    b0     = p_ls[nstim + 1]

    results = {}
    for name, nl_func, n_shape, p0_shape, bnds_shape in configs:
        p0     = np.concatenate([theta0, [gamma0, b0], p0_shape])
        bounds = [(None, None)] * (nstim + 2) + bnds_shape

        def _loss(params, _resp=resp, _sid=stim_ids, _pop=poprate,
                  _f=nl_func, _n=n_shape, _ns=nstim):
            u    = params[:_ns][_sid] + params[_ns] * _pop + params[_ns + 1]
            pred = _f(u, *params[_ns + 2: _ns + 2 + _n]) if _n else _f(u)
            return np.mean((_resp - pred) ** 2)

        try:
            opt   = minimize(_loss, p0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            theta = opt.x[:nstim]
            gamma = opt.x[nstim]
            b     = opt.x[nstim + 1]
            shape = list(opt.x[nstim + 2: nstim + 2 + n_shape]) if n_shape else []
            u     = theta[stim_ids] + gamma * poprate + b
            pred  = nl_func(u, *shape) if n_shape else nl_func(u)
            r2    = r2_score(resp, pred)
            results[name] = dict(r2=r2, theta=theta, gamma=gamma, b=b,
                                 nl_par=shape, pred=pred, u=u,
                                 stim_ids = stim_ids,poprate=poprate,
                                 resp_norm=resp, success=opt.success)
        except Exception:
            results[name] = dict(r2=np.nan, theta=None, gamma=None, b=None,
                                 nl_par=None, pred=None, u=None,
                                 stim_ids = None,poprate=None,
                                 resp_norm=resp, success=False)
    return results


# Fit all neurons across all sessions and collect R², Gamma, Beta, theta, nl_par
def fit_nl_models_sessions(sessions, nl_configs=NL_CONFIGS, verbose=False):
    nSessions = len(sessions)
    nl_names = [c[0] for c in nl_configs]
    nNL      = len(NL_CONFIGS)

    theta_arr  = {name: [] for name in nl_names}   # (nstim,) per neuron per model
    nlpar_arr  = {name: [] for name in nl_names}   # shape params per neuron per model
    ses_idx_arr = []                               # session index for each neuron

    for ises in range(nSessions):
        ses      = sessions[ises]
        ustim_s  = np.unique(ses.trialdata['Orientation'])
        stim_ids = np.searchsorted(ustim_s, ses.trialdata['Orientation'].to_numpy())
        N        = ses.respmat.shape[0]
        nstim    = len(ustim_s)

        if not hasattr(ses,'popratemat'):
            ses = comp_poprate(ses,version='radius_500')
            
        for name in nl_names:
            ses.celldata['R2'    + name] = np.nan
            ses.celldata['Gamma' + name] = np.nan
            ses.celldata['Beta'  + name] = np.nan

        for iN in tqdm(range(N), desc=f'Session {ises+1}/{nSessions}'):
            poprate = ses.popratemat[iN,:]
            resp = ses.respmat[iN, :]
            res  = fit_nl_models(resp, stim_ids, poprate, configs=nl_configs)
            for name in nl_names:
                ses.celldata.loc[iN, 'R2'    + name] = res[name]['r2']
                ses.celldata.loc[iN, 'Gamma' + name] = res[name]['gamma']
                ses.celldata.loc[iN, 'Beta'  + name] = res[name]['b']
                theta_arr[name].append(
                    res[name]['theta'] if res[name]['theta'] is not None
                    else np.full(nstim, np.nan))
                nlpar_arr[name].append(res[name]['nl_par'] or [])
            ses_idx_arr.append(ises)

    return sessions, theta_arr, nlpar_arr, ses_idx_arr

# Diagnostic figure for the example neuron
def diagnostic_nonlinfit(results_ex):
    pref_k    = int(np.argmax(results_ex[nl_names[0]]['theta']))
    # stim_ids??
    # oris = ses.trialdata['Orientation'].to_numpy()
    stim_ids = results_ex[nl_names[0]]['stim_ids']
    nstim    = len(np.unique(stim_ids))
    ustim    = np.linspace(0,360-360/nstim,nstim)
    poprate = results_ex[nl_names[0]]['poprate']

    orth_k    = (pref_k + nstim//4) % nstim
    pop_sweep = np.linspace(np.percentile(poprate, 1), np.percentile(poprate, 99), 200)

    best_name = max(nl_names, key=lambda n: results_ex[n]['r2']
                    if not np.isnan(results_ex[n]['r2']) else -1)
    best_res     = results_ex[best_name]
    resp_norm_ex = best_res['resp_norm']
    residuals    = resp_norm_ex - best_res['pred']

    fig, axes = plt.subplots(3, 3, figsize=(22*cm, 17*cm))

    # (0,0) Fitted nonlinearity shapes over the actual input range seen by each model
    ax = axes[0, 0]
    for i, (name, nl_func, n_shape, _, _) in enumerate(NL_CONFIGS):
        entry = results_ex[name]
        if entry['theta'] is None:
            continue
        u_vals  = entry['u']
        u_sweep = np.linspace(np.percentile(u_vals, 1), np.percentile(u_vals, 99), 300)
        y = nl_func(u_sweep, *entry['nl_par']) if n_shape else nl_func(u_sweep)
        u_norm = np.linspace(0,1,300)
        ax.plot(u_norm, y, color=clrs_nl[i], lw=2, label=name)
        # ax.plot(u_sweep, y, color=clrs_nl[i], lw=2, label=name)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('u  (θ_k + γ·P + b)')
    ax.set_ylabel('f(u)  [normalised scale]')
    ax.set_title('Fitted nonlinearities\n(over actual input range)')
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax, trim=True, offset=3)

    # (0,1) Fitted θ — tuning curve in input space
    ax = axes[0, 1]
    for i, (name, *_) in enumerate(NL_CONFIGS):
        if results_ex[name]['theta'] is None:
            continue
        # ax.plot(ustim, results_ex[name]['theta'], color=clrs_nl[i], lw=1.5,
        ax.plot(ustim, zscore(results_ex[name]['theta']), color=clrs_nl[i], lw=1.5,
                marker='o', ms=3, label=name)
    ax.set_xlabel('Orientation (°)')
    ax.set_ylabel('θ_k  (input-space drive)')
    ax.set_title('Fitted stimulus drive (pre-NL)')
    ax.legend(fontsize=7, frameon=False)
    ax.set_xticks(ustim[::2])
    ax.tick_params(axis='x', labelrotation=45)
    sns.despine(ax=ax, trim=True, offset=3)

    # (0,2) Mean output tuning curve: observed vs all model predictions
    ax = axes[0, 2]
    mean_obs = np.array([np.mean(resp_norm_ex[stim_ids == k]) for k in range(nstim)])
    ax.plot(ustim, mean_obs, color='k', lw=2, marker='o', ms=4, label='observed', zorder=5)
    for i, (name, nl_func, n_shape, _, _) in enumerate(NL_CONFIGS):
        entry = results_ex[name]
        if entry['pred'] is None:
            continue
        mean_pred = np.array([np.mean(entry['pred'][stim_ids == k]) for k in range(nstim)])
        ax.plot(ustim, mean_pred, color=clrs_nl[i], lw=1.5, ls='--', label=name)
    ax.set_xlabel('Orientation (°)')
    ax.set_ylabel('Mean response (normalised)')
    ax.set_title('Mean tuning curve\n(observed vs fitted)')
    ax.legend(fontsize=7, frameon=False)
    ax.set_xticks(ustim[::2])
    ax.tick_params(axis='x', labelrotation=45)
    sns.despine(ax=ax, trim=False, offset=3)

    # (1,0) Response vs pop rate for preferred and orthogonal orientations
    ax = axes[1, 0]
    for k_ori, lbl, col in [(pref_k, 'pref', 'tab:blue'), (orth_k, 'orth', 'tab:orange')]:
        idx_T = stim_ids == k_ori
        ax.scatter(poprate[idx_T], resp_norm_ex[idx_T], s=4, alpha=0.35, color=col,
                zorder=1, label=f'data ({lbl})')
        for i, (name, nl_func, n_shape, _, _) in enumerate(NL_CONFIGS):
            entry = results_ex[name]
            if entry['theta'] is None:
                continue
            u_line    = entry['theta'][k_ori] + entry['gamma'] * pop_sweep + entry['b']
            pred_line = nl_func(u_line, *entry['nl_par']) if n_shape else nl_func(u_line)
            ax.plot(pop_sweep, pred_line, color=clrs_nl[i], lw=1.2, alpha=0.8)
    ax.set_xlabel('Population rate (z)')
    ax.set_ylabel('Response (normalised)')
    ax.set_title('Resp vs pop rate\n(pref & orth, all models)')
    ax.set_xlim([pop_sweep[0], pop_sweep[-1]])
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax, trim=False, offset=3)

    # (1,1) R² bar plot
    ax = axes[1, 1]
    r2s = [results_ex[n]['r2'] for n in nl_names]
    ax.bar(np.arange(nNL), r2s, color=clrs_nl)
    ax.set_xticks(np.arange(nNL))
    ax.set_ylabel('R²')
    ax.set_title(f'R² per model')
    ax.set_ylim([0, max(r for r in r2s if not np.isnan(r)) * 1.25])
    for i, v in enumerate(r2s):
        if not np.isnan(v):
            ax.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    sns.despine(ax=ax, trim=False, offset=3)
    ax.set_xticklabels(nl_names, rotation=45, ha='right', fontsize=8)

    # (1,2) Predicted vs observed (best model, normalised space)
    ax = axes[1, 2]
    ax.scatter(resp_norm_ex, best_res['pred'], s=2, alpha=0.3, color='k')
    lims = [min(resp_norm_ex.min(), best_res['pred'].min()),
            max(resp_norm_ex.max(), best_res['pred'].max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('Observed (normalised)')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Predicted vs observed\n({best_name}, R²={best_res["r2"]:.3f})')
    sns.despine(ax=ax, trim=False, offset=3)

    # (2,0) Distribution of fitted inputs u across models
    ax = axes[2, 0]
    for i, (name, *_) in enumerate(NL_CONFIGS):
        u = results_ex[name]['u']
        if u is not None:
            sns.kdeplot(u, ax=ax, color=clrs_nl[i], label=name, fill=False)
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('Input  u = θ_k + γ·P + b')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of fitted inputs')
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax, trim=False, offset=3)

    # (2,1) Residuals vs pop rate (best model)
    ax = axes[2, 1]
    ax.scatter(poprate, residuals, s=2, alpha=0.3, color='k')
    ax.axhline(0, color='r', lw=1)
    _, _, rv, pv, _ = linregress(poprate, residuals)
    ax.text(0.05, 0.93, f'r={rv:.2f}, p={pv:.2e}', transform=ax.transAxes, fontsize=8)
    ax.set_xlabel('Population rate (z)')
    ax.set_ylabel('Residual')
    ax.set_title(f'Residuals vs pop rate  ({best_name})')
    sns.despine(ax=ax, trim=False, offset=3)

    # (2,2) Mean residuals per orientation (best model)
    ax = axes[2, 2]
    mean_resid = [np.mean(residuals[stim_ids == k]) for k in range(nstim)]
    ax.bar(ustim, mean_resid, width=18, color='steelblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Orientation (°)')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'Residuals by orientation  ({best_name})')
    ax.set_xticks(ustim[::2])
    ax.tick_params(axis='x', labelrotation=45)
    sns.despine(ax=ax, trim=False, offset=3)

    # plt.suptitle(f'NL model fits', fontsize=8, y=1.01)
    plt.tight_layout()

    return fig

# Analytical TF derivative functions  (match NL_CONFIGS exactly)
def tfd_linear(u):
    return np.ones_like(u)

def tfd_relu(u):
    return (u > 0).astype(float)

def tfd_softplus(u, beta):
    b = np.abs(beta) + 1e-4
    return 1.0 / (1.0 + np.exp(-np.clip(b * u, -500.0, 500.0)))

def tfd_sigmoid(u, a):
    s = 1.0 / (1.0 + np.exp(-np.clip(u, -500.0, 500.0)))
    return np.abs(a) * s * (1.0 - s)

def tfd_tanh(u, a):
    return np.abs(a) * 0.5 * (1.0 - np.tanh(u) ** 2)

def tfd_powerlaw(u, p):
    exp = np.abs(p) + 1e-4
    return exp * np.power(np.maximum(u, 1e-8), exp - 1.0) * (u > 0).astype(float)

def tfd_exp(u):
    return np.maximum(0.0, np.exp(np.clip(u, -500.0, 10.0)))

TF_DERIVS = {
    'Linear':          tfd_linear,
    'ReLU':            tfd_relu,
    'Softplus':        tfd_softplus,
    'Sigmoid':         tfd_sigmoid,
    'Tanh':            tfd_tanh,
    'Power-law (p=2)': tfd_powerlaw,
    'Exp':             tfd_exp,
}

