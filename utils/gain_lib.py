
#%% 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import zscore,pearsonr,spearmanr
from tqdm import tqdm
from scipy.stats import linregress

from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes

def plot_respmat(orientations, datasets, labels, prefori):
    data = datasets[0]
    poprate = np.nanmean(data,axis=0)

    sort_idx_trials     = np.lexsort((poprate, orientations))[::-1]
    gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])
    sort_idx_neurons    = np.lexsort((gain_weights, prefori))[::-1]

    fig,axes = plt.subplots(1, len(datasets),figsize=(3*len(datasets),4))
    if len(datasets) == 1:
        axes = [axes]
    for d,data in enumerate(datasets):
        ax = axes[d]
        data = data[sort_idx_neurons,:][:,sort_idx_trials]
        # ax.imshow(data,aspect='auto',vmin=0.1,vmax=0.5,cmap='magma')
        # ax.imshow(data,aspect='auto',vmin=np.percentile(datasets[0],20),vmax=np.percentile(datasets[0],90),cmap='magma')
        ax.imshow(data,aspect='auto',vmin=np.percentile(datasets[0],20),vmax=np.percentile(datasets[0],90),cmap='magma')
        ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
        ax.set_xticks([0,np.shape(data)[1]],labels=[0,np.shape(data)[1]],fontsize=7)
        ax.set_title(labels[d])
        ax.set_xlabel('Trial',fontsize=9)
        ax.set_ylabel('Neuron',fontsize=9)
        ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    return fig

def plot_tuned_response(orientations, datasets, labels):
    fig,axes = plt.subplots(1, len(datasets),figsize=(3*len(datasets),4))
    if len(datasets) == 1:
        axes = [axes]
    u_oris = np.unique(orientations)
    for d,data in enumerate(datasets):
        ax = axes[d]
        sm = np.array([np.mean(data[:, orientations == i], axis=1) for i in u_oris])
        if d == 0:
            idx = np.argsort(np.argmax(sm,axis=0))
        sm = sm[:,idx]
        ax.imshow(sm.T,aspect='auto',vmin=np.percentile(datasets[0],20),vmax=np.percentile(datasets[0],90),cmap='magma')
        ax.set_xticks(np.arange(len(u_oris)),labels=u_oris,fontsize=7)
        ax.set_yticks([0,np.shape(data)[0]],labels=[0,np.shape(data)[0]],fontsize=7)
        ax.set_title(labels[d])
        ax.set_xlabel('Orientation',fontsize=9)
        ax.set_ylabel('Neuron',fontsize=9)
        ax.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()

    return fig

def tuned_resp_model(data, stimuli):
    nstim = len(np.unique(stimuli))
    assert nstim == 9 or nstim == 16, 'There should be 9 or 16 unique stimuli, not %d' %nstim
    
    data_hat = np.zeros_like(data)

    for i,stim in enumerate(np.unique(stimuli)):
        data_hat[:,stimuli==stim] = np.mean(data[:, stimuli==stim], axis=1,keepdims=True)

    return data_hat

def pop_rate_gain_model(data, stimuli):
    poprate             = np.nanmean(data,axis=0)
    gain_weights        = np.array([np.corrcoef(poprate,data[n,:])[0,1] for n in range(data.shape[0])])
    gain_trials         = poprate - np.nanmean(data,axis=None)

    ustim,istimeses,stims  = np.unique(stimuli,return_index=True,return_inverse=True)
    nstim = len(ustim)
    assert nstim == 9 or nstim == 16, 'There should be 9 or 16 unique stimuli, not %d' %nstim
    
    # Calculate mean response per stimulus
    sm = np.array([np.mean(data[:,stims == i,], axis=1) for i in range(nstim)])

    if np.mean(poprate) < 1: 
        mfs         = np.arange(10,30,2)
    else:
        mfs         = np.arange(0,0.3,0.025)

    r2data      = []
    for imf,mf in enumerate(mfs):
        data_hat = np.empty_like(data)
        for i in range(nstim):
            data_hat[:,stims == i] = sm[i,:][:,np.newaxis] * (1 + np.outer(gain_weights * mf,gain_trials[stims == i]))

        r2data.append(r2_score(data,data_hat))
    
    mf = mfs[np.argmax(r2data)]
    data_hat = np.empty_like(data)
    for i in range(nstim):
        data_hat[:,stims == i] = sm[i,:][:,np.newaxis] * (1 + np.outer(gain_weights * mf,gain_trials[stims == i]))

    return data_hat

def comp_poprate(sessions,version='allfast'):

    for ises,ses in enumerate(sessions):
        resp                = zscore(ses.respmat,axis=1)

        N                   = len(ses.celldata)

        ses.popratemat      = np.full_like(resp, np.nan)

        if not hasattr(ses,'distmat_xyz') and 'xloc' in ses.celldata:
            # ses = compute_pairwise_anatomical_distance([ses])
            [ses] = compute_pairwise_anatomical_distance([sessions[ises]])
    
        for iN in range(N):
            
            if version == 'allfast':
                ses.popratemat = np.tile(np.nanmean(resp, axis=0), (N,1))
                break
            elif version == 'all':
                ses.popratemat[iN,:] = np.nanmean(resp[np.setdiff1d(np.arange(N),iN),:], axis=0)
            elif version == 'area':
                idx_N = ses.celldata['roi_name'] == ses.celldata['roi_name'][iN]
                idx_N[iN] = False
                ses.popratemat[iN,:] = np.nanmean(resp[idx_N,:], axis=0)
            elif version == 'plane':
                # ses.popratemat[iN,:] = poprate_planes[ses.celldata['plane_idx'][iN]]
                idx_N = ses.celldata['plane_idx'] == ses.celldata['plane_idx'][iN]
                idx_N[iN] = False
                ses.popratemat[iN,:] = np.nanmean(resp[idx_N,:], axis=0)
            elif version == 'radius_50':
                idx_N = ses.distmat_xyz[iN,:] < 50
                idx_N[iN] = False
                ses.popratemat[iN,:] = np.nanmean(resp[idx_N,:], axis=0)
            elif version == 'radius_100':
                idx_N = ses.distmat_xyz[iN,:] < 100
                idx_N[iN] = False
                ses.popratemat[iN,:] = np.nanmean(resp[idx_N,:], axis=0)
            elif version == 'radius_500':
                idx_N = ses.distmat_xyz[iN,:] < 500
                idx_N[iN] = False
                ses.popratemat[iN,:] = np.nanmean(resp[idx_N,:], axis=0)
            elif version == 'radius_1000':
                idx_N = ses.distmat_xyz[iN,:] < 1000
                idx_N[iN] = False
                ses.popratemat[iN,:] = np.nanmean(resp[idx_N,:], axis=0)
            elif version == 'random':
                ses.popratemat[iN,:] = np.random.randn(1,resp.shape[1])
            elif version == 'runspeed':
                ses.popratemat[iN,:] = ses.respmat_runspeed
            elif version == 'videome':
                ses.popratemat[iN,:] = ses.respmat_videome
    return sessions

def compute_pop_coupling(sessions,version='allfast'):
    for ises,ses in enumerate(sessions):
        resp        = zscore(ses.respmat,axis=1)

        if not hasattr(ses,'popratemat') or np.all(np.isnan(ses.popratemat)):
            [ses] = comp_poprate([ses],version=version)

        ses.celldata['pop_coupling']   = [np.corrcoef(resp[i,:],ses.popratemat[i,:])[0,1] for i in range(len(ses.celldata))]
    
    return sessions


def fitAffine_GR_singleneuron_full(sessions,modelversion='radius_500',recompute_poprate=True):
    if recompute_poprate:
        print('recomputing population rate\n')
        sessions = comp_poprate(sessions,version=modelversion)

    for ses in tqdm(sessions,desc='Fitting Single Neuron Affine Model',total=len(sessions)):

        ses.celldata['aff_r2_grfull'] = np.nan
        ses.celldata['aff_alpha_grfull'] = np.nan
        ses.celldata['aff_beta_grfull'] = np.nan
        ses.celldata['aff_offset_grfull'] = np.nan

        Y           = zscore(ses.respmat, axis=1)

        T           = copy.deepcopy(Y)

        trial_ori   = ses.trialdata['Orientation']
        oris        = np.sort(trial_ori.unique())

        ## Compute tuned response:
        for ori in oris:
            ori_idx     = np.where(ses.trialdata['Orientation']==ori)[0]
            temp        = np.mean(Y[:,ses.trialdata['Orientation']==ori],axis=1)
            T[:,ori_idx] = np.repeat(temp[:, np.newaxis], len(ori_idx), axis=1)

        N               = ses.respmat.shape[0]

        Y_hat           = np.full_like(ses.respmat, np.nan)

        for iN in range(N):
            # idx_N       = ses.distmat_xyz[iN,:] < radius
            # idx_N[iN]   = False
            r           = ses.popratemat[iN,:]
        
            y           = Y[iN,:]
            x           = T[iN,:]
            
            if np.isnan(r).all():
                # modelcoefs[modelversions.index(modelversion), iN, :] = np.nan
                # model_R2[modelversions.index(modelversion), iN] = np.nan
                # Y_hat[iN,:,modelversions.index(modelversion)] = np.nan
                continue
            # Construct the design matrix
            A = np.vstack([r * x, r, np.ones_like(y)]).T

            # Perform linear regression using least squares
            coefs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

            # Store the coefficients
            [ses.celldata.loc[iN,'aff_alpha_grfull'], ses.celldata.loc[iN,'aff_beta_grfull'], 
                ses.celldata.loc[iN,'aff_offset_grfull']] = coefs

            # Compute R^2 value
            y_pred = A @ coefs
            ses.celldata.loc[iN,'aff_r2_grfull'] = r2_score(y, y_pred)

    return sessions

def fitAffine_GR_singleneuron_split(sessions,modelversion='radius_500',perc=50,recompute_poprate=True):
    if recompute_poprate:
        sessions = comp_poprate(sessions,version=modelversion)

    for ses in tqdm(sessions,desc='Fitting Single Neuron Affine Model',total=len(sessions)):

        ses.celldata['aff_r2_grsplit'] = np.nan
        ses.celldata['aff_alpha_grsplit'] = np.nan
        ses.celldata['aff_beta_grsplit'] = np.nan

        trial_ori   = ses.trialdata['Orientation']
        oris        = np.sort(trial_ori.unique())

        N = ses.respmat.shape[0]

        Y_hat           = np.full_like(ses.respmat, np.nan)

        for iN in range(N):
            # idx_N       = ses.distmat_xyz[iN,:] < radius
            # idx_N[iN]   = False
            r           = ses.popratemat[iN,:]
            
            if np.isnan(r).all():
                # modelcoefs[modelversions.index(modelversion), iN, :] = np.nan
                # model_R2[modelversions.index(modelversion), iN] = np.nan
                # Y_hat[iN,:,modelversions.index(modelversion)] = np.nan
                continue

            idx_low    = r<=np.percentile(r,perc)
            idx_high   = r>np.percentile(r,100-perc)

            meanresp    = np.empty([len(oris),2])
            for i,ori in enumerate(oris):
                meanresp[i,0] = np.nanmean(ses.respmat[iN,np.logical_and(ses.trialdata['Orientation']==ori,idx_low)])
                meanresp[i,1] = np.nanmean(ses.respmat[iN,np.logical_and(ses.trialdata['Orientation']==ori,idx_high)])
                
            # meanresp_pref          = meanresp.copy()
            # for n in range(N):
            #     meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            #     meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

            # normalize by peak response during still trials
            tempmin,tempmax = meanresp[:,0].min(axis=0,keepdims=True),meanresp[:,0].max(axis=0,keepdims=True)
            meanresp[:,0] = (meanresp[:,0] - tempmin) / (tempmax - tempmin)
            meanresp[:,1] = (meanresp[:,1] - tempmin) / (tempmax - tempmin)
            
            b = linregress(meanresp[:,0],meanresp[:,1])

            # Store the coefficients
            [ses.celldata.loc[iN,'aff_alpha_grsplit'], ses.celldata.loc[iN,'aff_beta_grsplit'], 
                ses.celldata.loc[iN,'aff_r2_grsplit']] = b[:3]
            
    return sessions

def scatter_alphabeta(ax,celldata,xfield='aff_alpha_grsplit',yfield='aff_beta_grsplit'):
    sns.scatterplot(data=celldata,x=xfield,y=yfield,
                    color='green',ax=ax,hue='roi_name',marker='.',s=8)
    sns.regplot(data=celldata,x=xfield,y=yfield,
                color='k',line_kws={'linewidth': 1},scatter=False)
    print('r=%2.2g,p=%2.2g' % pearsonr(celldata[xfield],celldata[yfield]))
    ax.set_xlabel('mult')
    ax.set_ylabel('add')
    sns.despine(ax=ax,top=True,right=True,offset=2)
    