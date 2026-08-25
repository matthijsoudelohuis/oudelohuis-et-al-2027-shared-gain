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
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_B_2_all' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%% Make the 3D PCA figure:
ises = 1
ses = sessions[ises]

# idx_N = sessions[ises].celldata['gOSI']>0
# idx_N = sessions[ises].celldata['tuning_var']>0
idx_N = np.all((
                ses.celldata['roi_name']=='V2',
                # ses.celldata['roi_name']=='V2',
                # ses.celldata['gOSI']>0.3,
                # ses.celldata['tuning_var']>0.1
                ),axis=0)
fig = plot_PCA_gratings_3D(sessions[ises],idx_N=idx_N,size='poprate')
axes = fig.get_axes()
axes[0].view_init(elev=-45, azim=0, roll=-10)
axes[0].set_zlim([-5,45])
# fig.savefig(os.path.join(figdir,'Cone_3D_V1_Original_%s' % sessions[0].sessiondata['session_id'][0] + '.png'), format = 'png')

#%% ######### plot result as scatter by orientation for projection ########
ises = 1
ses = sessions[ises]
idx_N = np.all((
                # ses.celldata['roi_name']=='V1',
                ses.celldata['roi_name']=='V2',
                # ses.celldata['gOSI']>0.3,
                # ses.celldata['tuning_var']>0.025
                ),axis=0)

respmat         = ses.respmat[idx_N, :]
# respmat         = zscore(ses.respmat[idx_N, :],axis=1)
poprate         = np.nanmean(respmat,axis=0)

#Get the population rate / gain axis:
G        = np.ones(np.sum(idx_N)) #np.clip(gain_weights,0,1)

oris    = np.mod(ses.trialdata['Orientation'],180)
uoris   = np.sort(np.unique(oris))
noris   = len(uoris)

ex_ori      = 0
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
