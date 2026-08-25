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
# import seaborn as sns
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.stats import zscore,wilcoxon

# from loaddata.session_info import filter_sessions
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import *
# from utils.CCAlib import *
# from utils.corr_lib import *
# from utils.tuning import compute_tuning_wrapper
# from utils.regress_lib import *
# from utils.gain_lib import *
from utils.params import params

figdir = os.path.join(params['figdir'],'popgeometry')

#%% 
#        #####     #    ######     ######     #    #######    #    
#       #     #   # #   #     #    #     #   # #      #      # #   
#       #     #  #   #  #     #    #     #  #   #     #     #   #  
#       #     # #     # #     #    #     # #     #    #    #     # 
#       #     # ####### #     #    #     # #######    #    ####### 
#       #     # #     # #     #    #     # #     #    #    #     # 
#######  #####  #     # ######     ######  #     #    #    #     # 

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_B_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_D_2_all' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_B_2_all' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_E_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Make the 3D PCA figure:
ises = 0
ses = sessions[ises]

# idx_N = sessions[ises].celldata['gOSI']>0
# idx_N = sessions[ises].celldata['tuning_var']>0
idx_N = np.all((
                # ses.celldata['roi_name']=='V1',
                # ses.celldata['roi_name']=='V2',
                # ses.celldata['gOSI']>0.3,
                ses.celldata['tuning_var']>0.025,
                # np.mean(ses.respmat,axis=1)<4,
                # np.mean(ses.respmat,axis=1)>4,
                ),axis=0)
fig = plot_PCA_gratings_3D(sessions[ises],idx_N=idx_N,size='poprate')
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])
# fig.savefig(os.path.join(figdir,'Cone_3D_V1_Original_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ######### plot result as scatter by orientation for projection ########
ises = 0
ses = sessions[ises]
idx_N = np.all((
                # ses.celldata['roi_name']=='V1',
                # ses.celldata['roi_name']=='V2',
                # ses.celldata['roi_name']=='LI',
                # ses.celldata['roi_name']=='PM',
                # ses.celldata['gOSI']>0.1,
                ses.celldata['tuning_var']>0.0,
                # np.mean(ses.respmat,axis=1)>3,
                # np.mean(ses.respmat,axis=1)>3,
                # np.mean(ses.respmat,axis=1)<8,
                ),axis=0)

respmat         = ses.respmat[idx_N, :]
# respmat         = zscore(ses.respmat[idx_N, :],axis=1)
poprate         = np.nanmean(respmat,axis=0)

#Get the population rate / gain axis:
G        = np.ones(np.sum(idx_N)) #np.clip(gain_weights,0,1)

oris    = np.mod(ses.trialdata['Orientation'],180)
# oris    = ses.trialdata['Orientation']
uoris   = np.sort(np.unique(oris))
noris   = len(uoris)

ex_ori      = 30
# ex_ori = 135
ex_ori = uoris[0]
# idx_T = oris==ex_ori
idx_T = np.logical_and(oris==ex_ori,ses.trialdata['Contrast']==0.1) 

#Get the population vector for a specific orientation:
W_ori = np.nanmean(respmat[:,idx_T],axis=1)

# Make W_ori orthogonal to G with QR decomposition
Q, R = np.linalg.qr(np.column_stack([G, W_ori]))
W_ori = Q[:, 1] * R[1, 1]

#Show that the weight vector for this specific orientation is aligned with the preferred orientation of neurons:
# plt.scatter(ses.celldata['pref_ori'][idx_N],W_ori)

#Get the projection of the population data onto the gain axis and the orientation axis:
X_gain = G @ respmat[:,idx_T]
Y_ori = W_ori @ respmat[:,idx_T]

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
# my_savefig(fig,figdir,'Cone_Slice_V1_%s_alloris' % (ses.session_id))

#%% Fit linear and exponential curves to the gain axis vs orientation axis projection,
# for each orientation, for each session, and store the R2 of both fits:
oris_allses     = [np.mod(ses.trialdata['Orientation'],180) for ses in sessions]
uoris           = np.unique(np.round(np.concatenate(oris_allses),3))
nStim           = len(uoris)

r2_mat          = np.full((3,nSessions,nStim),np.nan) #dim0: 0=linear, 1=exponential, 2=log

for ises,ses in enumerate(sessions):
    # No cell subselection: use every neuron in the session.
    # idx_N = np.ones(ses.respmat.shape[0],dtype=bool)
    idx_N = np.all((
                ses.celldata['roi_name']=='V1',
                # ses.celldata['roi_name']=='V2',
                # ses.celldata['gOSI']>0.5,
                # ses.celldata['tuning_var']<0.3,
                # ses.celldata['tuning_var']>0.3,
                # np.mean(ses.respmat,axis=1)>3,
                # np.mean(ses.respmat,axis=1)>4
                ),axis=0)
    
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

# Show paired R2 of linear vs exponential fit as an xy scatterplot with a paired Wilcoxon test:
r2_lin = r2_mat[0].flatten()
# r2_exp = r2_mat[1].flatten()
r2_exp = r2_mat[2].flatten()
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
# my_savefig(fig,figdir,'R2_Linear_vs_Exponential_V1_%dsessions' % nSessions)
# my_savefig(fig,figdir,'R2_Linear_vs_Logarithmic_V1_%dsessions' % nSessions)
