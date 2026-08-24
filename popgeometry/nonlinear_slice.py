
#%% Import libs:
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore,wilcoxon
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.plot_lib import * 
from utils.params import params

figdir = os.path.join(params['figdir'],'popgeometry','curvature')

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_A_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% ########################### Compute tuning metrics: ###################################
# sessions = compute_tuning_wrapper(sessions)
# sessions = compute_pop_coupling(sessions,version='radius_500')

#%% ######### plot result as scatter by orientation for projection ########
ses = sessions[0]
idx_N = np.all((
                ses.celldata['roi_name']=='V1',
                ses.celldata['gOSI']>0.5,
                # ses.celldata['tuning_var']>0.025
                ),axis=0)

# zscore for each neuron across trial responses
respmat         = ses.respmat[idx_N, :]
poprate         = np.nanmean(respmat,axis=0)

#Get the population rate / gain axis:
G        = np.ones(np.sum(idx_N)) #np.clip(gain_weights,0,1)


oris = np.mod(ses.trialdata['Orientation'],180)
uoris = np.sort(np.unique(oris))
noris = len(uoris)

ex_ori = 90
# ex_ori = 135
# ex_ori = 67.5
#Get the population vector for a specific orientation:
W_ori = np.nanmean(respmat[:,oris==ex_ori],axis=1)

# Make W_ori orthogonal to G with QR decomposition
Q, R = np.linalg.qr(np.column_stack([G, W_ori]))
W_ori = Q[:, 1] * R[1, 1]

#Show that the weight vector for this specific orientation is aligned with the preferred orientation of neurons:
# plt.scatter(ses.celldata['pref_ori'][idx_N],W_ori)

#Get the projection of the population data onto the gain axis and the orientation axis:
X_gain = G @ respmat[:,oris==ex_ori]
Y_ori = W_ori @ respmat[:,oris==ex_ori]

fig,axes = plt.subplots(1,1,figsize=[4*cm, 4*cm])
ax = axes
ax.scatter(X_gain,Y_ori,s=1,alpha=0.5,color='k')
ax.set_xlabel('projection onto\npop. rate axis')
ax.set_ylabel('orientation axis')

# Fit linear curve
z_lin = np.polyfit(X_gain, Y_ori, 1)
p_lin = np.poly1d(z_lin)
X_sort = np.sort(X_gain)
y_lin_pred = p_lin(X_sort)
ss_res_lin = np.sum((Y_ori - p_lin(X_gain))**2)
ss_tot = np.sum((Y_ori - np.mean(Y_ori))**2)
r2_lin = 1 - (ss_res_lin / ss_tot)

# Fit exponential curve
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit logarithmic curve (with a positive offset to avoid log(0))
def log_func(x, a, b, c):
    return a * np.log(np.abs(x) + b + 1e-6) + c

try:
    x_range = np.max(X_gain) - np.min(X_gain)
    p0 = [np.max(Y_ori) - np.min(Y_ori), -2 / x_range, np.min(Y_ori)]
    popt, _ = curve_fit(exp_func, X_gain, Y_ori, p0=p0, maxfev=5000)
    y_exp_pred = exp_func(X_sort, *popt)
    ss_res_exp = np.sum((Y_ori - exp_func(X_gain, *popt))**2)
    r2_exp = 1 - (ss_res_exp / ss_tot)
    ax.plot(X_sort, y_exp_pred, 'r-', linewidth=1, label=f'Exp (R² = {r2_exp:.3f})')
except Exception:
    r2_exp = np.nan

try:
    x_range = np.max(X_gain) - np.min(X_gain)
    x_offset = np.max([x_range * 0.1, 1e-3])
    p0 = [np.max(Y_ori) - np.min(Y_ori), x_offset, np.min(Y_ori)]
    popt, _ = curve_fit(log_func, X_gain, Y_ori, p0=p0, maxfev=5000)
    y_log_pred = log_func(X_sort, *popt)
    ss_res_log = np.sum((Y_ori - log_func(X_gain, *popt))**2)
    r2_log = 1 - (ss_res_log / ss_tot)
    ax.plot(X_sort, y_log_pred, 'g-', linewidth=1, label=f'Log (R² = {r2_log:.3f})')
except Exception:
    r2_log = np.nan

ax.plot(X_sort, y_lin_pred, 'b-', linewidth=1, label=f'Linear (R² = {r2_lin:.3f})')
ax.legend(fontsize=5, frameon=False)

sns.despine(fig=fig,top=True,right=True,offset=2)
# my_savefig(fig,figdir,'Cone_Slice_V1_%s_ori_%d' % (ses.session_id, ex_ori))

#%% 
pal         = np.tile(sns.color_palette('husl', 8), (2, 1))
fit_version = 'log'

fig,axes = plt.subplots(1,noris,figsize=[3*noris*cm, 3*cm],sharex=True,sharey=True)
for iori,ori in enumerate(uoris):
    ax = axes[iori]
    #Get the population vector for a specific orientation:
    W_ori = np.nanmean(respmat[:,oris==ori],axis=1)

    #Get the projection of the population data onto the gain axis and the orientation axis:
    X_gain = G @ respmat[:,oris==ori]
    Y_ori = W_ori @ respmat[:,oris==ori]

    ax.scatter(X_gain,Y_ori,s=1,alpha=0.5,color=pal[iori])
    if iori == noris//2:
        ax.set_xlabel('projection onto pop. rate axis')
    if iori == 0:
        ax.set_ylabel('projection onto\norientation axis')

    if fit_version == 'exp': 
        try:
            x_range = np.max(X_gain) - np.min(X_gain)
            p0 = [np.max(Y_ori) - np.min(Y_ori), -2.0 / x_range, np.min(Y_ori)]
            popt, _ = curve_fit(exp_func, X_gain, Y_ori, p0=p0, maxfev=5000)
            y_exp_pred = exp_func(X_sort, *popt)
            ss_res_exp = np.sum((Y_ori - exp_func(X_gain, *popt))**2)
            r2_exp = 1 - (ss_res_exp / ss_tot)
            ax.plot(X_sort, y_exp_pred, 'k-', linewidth=1)
        except:
            r2_exp = np.nan

    elif fit_version == 'log':
        try:
            x_range = np.max(X_gain) - np.min(X_gain)
            x_offset = np.max([x_range * 0.1, 1e-3])
            p0 = [np.max(Y_ori) - np.min(Y_ori), x_offset, np.min(Y_ori)]
            popt, _ = curve_fit(log_func, X_gain, Y_ori, p0=p0, maxfev=5000)
            y_log_pred = log_func(X_sort, *popt)
            ss_res_log = np.sum((Y_ori - log_func(X_gain, *popt))**2)
            r2_log = 1 - (ss_res_log / ss_tot)
            ax.plot(X_sort, y_log_pred, 'k-', linewidth=1)
        except Exception:
            r2_log = np.nan
                            
    ax.set_title('%.1f (deg)' % ori,fontsize=7)
sns.despine(fig=fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'Cone_Slice_V1_%s_alloris' % (ses.session_id))


#%% Load all sessions:
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_A_2_V1only' + '.npz'),allow_pickle=True)
sessions = data['sessions']
nSessions = len(sessions)

sessions = compute_tuning_wrapper(sessions)
sessions = compute_pop_coupling(sessions,version='radius_500')

#%% Fit linear and exponential curves to the gain axis vs orientation axis projection,
# for each orientation, for each session, and store the R2 of both fits:
oris_allses     = [np.mod(ses.trialdata['Orientation'],180) for ses in sessions]
uoris           = np.unique(np.round(np.concatenate(oris_allses),3))
nStim           = len(uoris)

r2_mat          = np.full((3,nSessions,nStim),np.nan) #dim0: 0=linear, 1=exponential

for ises,ses in enumerate(sessions):
    # No cell subselection: use every neuron in the session.
    idx_N = np.ones(ses.respmat.shape[0],dtype=bool)
    if np.sum(idx_N) < 5:
        continue

    respmat     = ses.respmat[idx_N,:]
    G           = np.ones(np.sum(idx_N))
    oris        = np.mod(ses.trialdata['Orientation'],180)

    for iori,ori in enumerate(uoris):
        idx_T = np.isclose(oris,ori)
        if np.sum(idx_T) < 10:
            continue

        #Get the population vector for this orientation and orthogonalize to the gain axis:
        W_ori = np.nanmean(respmat[:,idx_T],axis=1)
        Q, R = np.linalg.qr(np.column_stack([G, W_ori]))
        W_ori = Q[:, 1] * R[1, 1]

        #Get the projection of the population data onto the gain axis and the orientation axis:
        X_gain = G @ respmat[:,idx_T]
        Y_ori = W_ori @ respmat[:,idx_T]

        ss_tot = np.sum((Y_ori - np.mean(Y_ori))**2)
        if ss_tot == 0:
            continue

        # Fit linear curve
        z_lin = np.polyfit(X_gain, Y_ori, 1)
        p_lin = np.poly1d(z_lin)
        ss_res_lin = np.sum((Y_ori - p_lin(X_gain))**2)
        r2_mat[0,ises,iori] = 1 - (ss_res_lin / ss_tot)

        # Fit exponential curve
        try:
            x_range = np.max(X_gain) - np.min(X_gain)
            p0 = [np.max(Y_ori) - np.min(Y_ori), -2.0 / x_range, np.min(Y_ori)]
            popt, _ = curve_fit(exp_func, X_gain, Y_ori, p0=p0, maxfev=5000)
            ss_res_exp = np.sum((Y_ori - exp_func(X_gain, *popt))**2)
            r2_mat[1,ises,iori] = 1 - (ss_res_exp / ss_tot)
        except Exception:
            pass
        # Fit logarithmic curve
        try:
            x_range = np.max(X_gain) - np.min(X_gain)
            x_offset = np.max([x_range * 0.1, 1e-3])
            p0 = [np.max(Y_ori) - np.min(Y_ori), x_offset, np.min(Y_ori)]
            popt, _ = curve_fit(log_func, X_gain, Y_ori, p0=p0, maxfev=5000)
            y_log_pred = log_func(X_sort, *popt)
            ss_res_log = np.sum((Y_ori - log_func(X_gain, *popt))**2)
            r2_mat[2,ises,iori] = 1 - (ss_res_log / ss_tot)
        except Exception:
            pass

#%% Show paired R2 of linear vs exponential fit as an xy scatterplot with a paired Wilcoxon test:
r2_lin = r2_mat[0].flatten()
r2_exp = r2_mat[1].flatten()
idx_valid = ~np.isnan(r2_lin) & ~np.isnan(r2_exp)
r2_lin = r2_lin[idx_valid]
r2_exp = r2_exp[idx_valid]
npairs = len(r2_lin)

stat,pval = wilcoxon(r2_lin, r2_exp)
print('Wilcoxon signed-rank test (linear vs exponential R2): W = %.1f, p = %.3e (n = %d session-orientation pairs)'
      % (stat, pval, npairs))

fig,ax = plt.subplots(1,1,figsize=[3.5*cm, 3.5*cm])
ax.scatter(r2_lin, r2_exp, s=6, alpha=0.6, color='r',edgecolor=None,marker='.')

# Show the equality line: points above it favor the exponential fit.
plot_min = my_floor(np.min(r2_lin),-1)
plot_max = 1
ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=0.8)
ax.set_xlim(plot_min, plot_max)
ax.set_ylim(plot_min, plot_max)
ax_nticks(ax,4)
ax.set_aspect('equal', adjustable='box')

# Annotate the paired Wilcoxon test result.
ptext = 'p = %2.2g' % pval if pval < 0.001 else 'p = %.3f' % pval
ax.text(0.05, 0.95, ptext, transform=ax.transAxes,
        ha='left', va='top', fontsize=6)
ax.set_xlabel('Linear fit R²')
ax.set_ylabel('Exponential fit R²')
sns.despine(fig=fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'R2_Linear_vs_Exponential_V1_%dsessions' % nSessions)

#%% Recompute fits for the bottom and top 20%% population-coupled neurons:
def compute_group_r2(percentile,top=False):
    group_r2 = np.full((2,nSessions,nStim),np.nan)
    for ises,ses in enumerate(sessions):
        coupling = np.asarray(ses.celldata['pop_coupling'])
        threshold = np.nanpercentile(coupling,percentile)
        idx_N = coupling >= threshold if top else coupling <= threshold
        if np.sum(idx_N) < 5:
            continue
        respmat = ses.respmat[idx_N,:]
        G = np.ones(np.sum(idx_N))
        oris = np.mod(ses.trialdata['Orientation'],180)
        for iori,ori in enumerate(uoris):
            idx_T = np.isclose(oris,ori)
            if np.sum(idx_T) < 10:
                continue
            W_ori = np.nanmean(respmat[:,idx_T],axis=1)
            Q,R = np.linalg.qr(np.column_stack([G,W_ori]))
            W_ori = Q[:,1] * R[1,1]
            X_gain = G @ respmat[:,idx_T]
            Y_ori = W_ori @ respmat[:,idx_T]
            ss_tot = np.sum((Y_ori-np.mean(Y_ori))**2)
            if ss_tot == 0:
                continue
            p_lin = np.poly1d(np.polyfit(X_gain,Y_ori,1))
            group_r2[0,ises,iori] = 1-np.sum((Y_ori-p_lin(X_gain))**2)/ss_tot
            try:
                x_range = np.ptp(X_gain)
                if x_range == 0:
                    continue
                p0 = [np.ptp(Y_ori),-2.0/x_range,np.min(Y_ori)]
                popt,_ = curve_fit(exp_func,X_gain,Y_ori,p0=p0,maxfev=5000)
                group_r2[1,ises,iori] = 1-np.sum((Y_ori-exp_func(X_gain,*popt))**2)/ss_tot
            except Exception:
                pass
    return group_r2

r2_bottom = compute_group_r2(20,top=False)
r2_top = compute_group_r2(80,top=True)
fig,ax = plt.subplots(1,1,figsize=[3.5*cm,3.5*cm])
for group,color,label in ((r2_bottom,'b','bottom 20%'),(r2_top,'r','top 20%')):
    x,y = group[0].flatten(),group[1].flatten()
    valid = ~np.isnan(x) & ~np.isnan(y)
    ax.scatter(x[valid],y[valid],s=6,alpha=0.6,color=color,edgecolor=None,marker='.',label=label)
plot_min = my_floor(np.nanmin(r2_bottom[0]),-1)
ax.plot([plot_min,1],[plot_min,1],'k--',linewidth=0.8)
ax.set_xlim(plot_min,1); ax.set_ylim(plot_min,1)
ax_nticks(ax,4); ax.set_aspect('equal',adjustable='box')
ax.set_xlabel('Linear fit R²'); ax.set_ylabel('Exponential fit R²')
ax.legend(frameon=False,fontsize=5,loc='upper left',title='pop. coupling')
sns.despine(fig=fig,top=True,right=True,offset=2)
my_savefig(fig,figdir,'R2_Linear_vs_Exponential_Top_Bottom20_%dsessions' % nSessions)
