"""
This script contains functions to compute noise correlations
on simultaneously acquired calcium imaging data with mesoscope
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

## Import libs:
import copy
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic,binned_statistic_2d
from skimage.measure import block_reduce
from tqdm import tqdm
import matplotlib.pyplot as plt
#Repeated measures ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

from utils.plot_lib import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.tuning import mean_resp_gn,mean_resp_gr,mean_resp_image 
from utils.rf_lib import filter_nearlabeled
from utils.pair_lib import *
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from utils.gain_lib import pop_rate_gain_model
import scipy.stats as ss
from scipy.optimize import curve_fit
from utils.shuffle_lib import * 

 #####  ####### #     # ######  #     # ####### #######     #####  ####### ######  ######  
#     # #     # ##   ## #     # #     #    #    #          #     # #     # #     # #     # 
#       #     # # # # # #     # #     #    #    #          #       #     # #     # #     # 
#       #     # #  #  # ######  #     #    #    #####      #       #     # ######  ######  
#       #     # #     # #       #     #    #    #          #       #     # #   #   #   #   
#     # #     # #     # #       #     #    #    #          #     # #     # #    #  #    #  
 #####  ####### #     # #        #####     #    #######     #####  ####### #     # #     # 

def compute_trace_correlation(sessions,uppertriangular=True,binwidth=1):
    """
    Compute the trace correlation between the calcium traces of all neurons in a session
    Trace correlation is computed by taking the mean of the fluorescence traces over a specified time window (binwidth)
    Parameters
    sessions : Session
        list of Session objects
    uppertriangular : bool
        if set to True, only upper triangular part of the correlation matrix is computed
    binwidth : float
        time window over which to compute the mean of the fluorescence trace
    Returns sessions
    """

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing trace correlations: '):
    
        avg_nframes     = int(np.round(sessions[ises].sessiondata['fs'][0] * binwidth))

        if avg_nframes > 1:
            arr_reduced     = block_reduce(sessions[ises].calciumdata.T, block_size=(1,avg_nframes), func=np.mean, cval=np.mean(sessions[ises].calciumdata.T))
        else:
            arr_reduced     = sessions[ises].calciumdata.T.to_numpy()

        sessions[ises].trace_corr                   = np.corrcoef(arr_reduced)

        N           = np.shape(sessions[ises].calciumdata)[1] #get dimensions of response matrix

        idx_triu    = np.tri(N,N,k=0)==1 #index only upper triangular part
        
        if uppertriangular:
            sessions[ises].trace_corr[idx_triu] = np.nan
        else:
            np.fill_diagonal(sessions[ises].trace_corr,np.nan)

        assert np.all(sessions[ises].trace_corr[~idx_triu] > -1)
        assert np.all(sessions[ises].trace_corr[~idx_triu] < 1)
    return sessions    

def compute_signal_noise_correlation(sessions,uppertriangular=True,filter_stationary=False,remove_method=None,remove_rank=0):
    # computing the pairwise correlation of activity that is shared due to mean response (signal correlation)
    # or residual to any stimuli in GR and GN protocols (noise correlation).

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing signal and noise correlations: '):
        if sessions[ises].sessiondata['protocol'][0]=='IM':
            [respmean,imageids]         = mean_resp_image(sessions[ises])
            [N,K]                       = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            sessions[ises].sig_corr     = np.corrcoef(respmean)

            if np.any(sessions[ises].trialdata['ImageNumber'].value_counts()>2):
                stims = sessions[ises].trialdata['ImageNumber'].to_numpy()
                idx = sessions[ises].trialdata['ImageNumber'].value_counts().index
                ustim = idx[np.where(sessions[ises].trialdata['ImageNumber'].value_counts()>2)[0]]
                
                # noise_corr = np.empty((N,N,len(ustim)))
                # for istim,stim in enumerate(ustim):
                #     respmat_res             = sessions[ises].respmat[:,stims==stim]
                #     respmat_res             -= np.nanmean(respmat_res,axis=1,keepdims=True)
                #     noise_corr[:,:,istim]   = np.corrcoef(respmat_res)

                respmat_res = np.full((N,K),np.nan)
                for istim,stim in enumerate(ustim):
                    temp                    = sessions[ises].respmat[:,stims==stim]
                    respmat_res[:,stims==stim]   = temp - np.nanmean(temp,axis=1,keepdims=True)
                respmat_res = respmat_res[:,~np.isnan(respmat_res).all(axis=0)]
                sessions[ises].noise_corr       = np.corrcoef(respmat_res)
            else:
                sessions[ises].noise_corr = np.full((np.shape(sessions[ises].sig_corr)),np.nan)
            
            if uppertriangular:
                idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].noise_corr[idx_triu] = np.nan
            else: #set only autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

        elif sessions[ises].sessiondata['protocol'][0]=='GR':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(sessions[ises].trialdata['Orientation'].unique())
            trialfilter                     = sessions[ises].respmat_runspeed<2 if filter_stationary else np.ones(K,bool)
            resp_meanori,respmat_res        = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
            prefori                         = oris[np.argmax(resp_meanori,axis=1)]

            sessions[ises].delta_pref       = np.abs(np.mod(np.subtract.outer(prefori, prefori),180))
            
            # Compute signal correlations on all trials: 
            # sessions[ises].sig_corr         = np.corrcoef(resp_meanori)

            #Compute signal correlation on separate halfs of trials:
            trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
            resp_meanori1,_                 = mean_resp_gr(sessions[ises],trialfilter=trialfilter)
            resp_meanori2,_                 = mean_resp_gr(sessions[ises],trialfilter=~trialfilter)
            sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_meanori1, resp_meanori2)[:N, N:] +
                                                np.corrcoef(resp_meanori2, resp_meanori1)[:N, N:])

            # plt.imshow(sessions[ises].sig_corr,vmin=-0.4,vmax=0.4)

            if remove_method is not None:
                if remove_method in ['PCA','FA','RRR']:

                    assert remove_rank > 0, 'remove_rank must be > 0'	
                    
                    trial_ori   = sessions[ises].trialdata['Orientation']
                    respmat_res = copy.deepcopy(sessions[ises].respmat)
                    respmat_res = zscore(respmat_res,axis=1)
                    
                    # for iarea,area in enumerate(sessions[ises].celldata['roi_name'].unique()):
                    #     idx = sessions[ises].celldata['roi_name'] == area
                    #     data = respmat_res[idx,:]

                        # data_hat = remove_dim(data,remove_method,remove_rank)

                    #     #Remove low rank prediction from data:
                    #     respmat_res[idx,:] = data - data_hat
                    
                    for i,ori in enumerate(oris):
                        data = respmat_res[:,trial_ori==ori]
                        
                        data_hat = remove_dim(data,remove_method,remove_rank)
                        
                        #Remove low rank prediction from data:
                        respmat_res[:,trial_ori==ori] = data - data_hat
                elif remove_method == 'GM':
                    stimuli         = np.array(sessions[ises].trialdata['stimCond'])
                    data_hat        = pop_rate_gain_model(sessions[ises].respmat, stimuli)
                    respmat_res     = sessions[ises].respmat - data_hat

            # Compute noise correlations from residuals:
            # sessions[ises].noise_corr       = np.corrcoef(respmat_res)
            # Compute per stimulus, then average:
            trial_ori   = sessions[ises].trialdata['Orientation']
            noise_corr = np.empty((N,N,len(oris)))  
            for i,ori in enumerate(oris):
                noise_corr[:,:,i] = np.corrcoef(respmat_res[:,trial_ori==ori])
            sessions[ises].noise_corr       = np.mean(noise_corr,axis=2)

            idx_triu = np.tri(N,N,k=0)==1 #index only upper triangular part
            if uppertriangular:
                sessions[ises].noise_corr[idx_triu] = np.nan
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].delta_pref[idx_triu] = np.nan
            else: #set only autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].delta_pref,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
        
        elif sessions[ises].sessiondata['protocol'][0]=='GN':
            [N,K]                           = np.shape(sessions[ises].respmat) #get dimensions of response matrix
            oris                            = np.sort(pd.Series.unique(sessions[ises].trialdata['centerOrientation']))
            speeds                          = np.sort(pd.Series.unique(sessions[ises].trialdata['centerSpeed']))
            trialfilter                     = sessions[ises].respmat_runspeed<2 if filter_stationary else np.ones(K,bool)
            resp_mean,respmat_res           = mean_resp_gn(sessions[ises],trialfilter)
            prefori, prefspeed              = np.unravel_index(resp_mean.reshape(N,-1).argmax(axis=1), (len(oris), len(speeds)))
            sessions[ises].prefori          = oris[prefori]
            sessions[ises].prefspeed        = speeds[prefspeed]

            # Compute signal correlations on all trials: 
            # sessions[ises].sig_corr         = np.corrcoef(resp_mean.reshape(N,len(oris)*len(speeds)))
            
            #Compute signal correlation on separate halfs of trials:
            trialfilter                     = np.random.choice([True,False],size=(K),p=[0.5,0.5])
            resp_mean1,_                    = mean_resp_gn(sessions[ises],trialfilter = trialfilter)
            resp_mean2,_                    = mean_resp_gn(sessions[ises],trialfilter = ~trialfilter)
            # sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_mean1, resp_mean2)[:N, N:] +
                                                # np.corrcoef(resp_mean2, resp_mean1)[:N, N:])
            sessions[ises].sig_corr         = 0.5 * (np.corrcoef(resp_mean1.reshape(N,-1), resp_mean2.reshape(N,-1))[:N, N:] +
                                                np.corrcoef(resp_mean2.reshape(N,-1), resp_mean1.reshape(N,-1))[:N, N:])
            if remove_method is not None:
                if remove_method in ['PCA','FA','RRR']:
                    assert remove_rank > 0, 'remove_rank must be > 0'	
                    respmat_res = copy.deepcopy(sessions[ises].respmat)
                    respmat_res = zscore(respmat_res,axis=1)

                    trial_ori   = sessions[ises].trialdata['centerOrientation']
                    trial_spd   = sessions[ises].trialdata['centerSpeed']
                    for iO,ori in enumerate(oris):
                        for iS,speed in enumerate(speeds):
                            idx_trial = np.logical_and(trial_ori==ori,trial_spd==speed)
                            data = respmat_res[:,idx_trial]
                            data_hat = remove_dim(data,remove_method,remove_rank)
                            #Remove low rank prediction from data:
                            respmat_res[:,idx_trial] = data - data_hat
                elif remove_method == 'GM':
                    stimuli         = np.array(sessions[ises].trialdata['stimCond'])
                    data_hat        = pop_rate_gain_model(sessions[ises].respmat, stimuli)
                    respmat_res     = sessions[ises].respmat - data_hat

            # Detrend the data:
            # respmat_res = detrend(respmat_res,axis=1)

            #Compute noise correlations from residuals:
            sessions[ises].noise_corr       = np.corrcoef(respmat_res)

            idx_triu = np.tri(N,N,k=0)==1   #index upper triangular part
            if uppertriangular:
                sessions[ises].sig_corr[idx_triu] = np.nan
                sessions[ises].noise_corr[idx_triu] = np.nan
            else: #set autocorrelation to nan
                np.fill_diagonal(sessions[ises].sig_corr,np.nan)
                np.fill_diagonal(sessions[ises].noise_corr,np.nan)

            assert np.all(sessions[ises].sig_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].sig_corr[~idx_triu] < 1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] > -1)
            assert np.all(sessions[ises].noise_corr[~idx_triu] < 1)
        # else, do nothing, skipping protocol other than GR, GN, and IM'

    return sessions

#     # ###  #####  #######     #####  ####### ######  ######  
#     #  #  #     #    #       #     # #     # #     # #     # 
#     #  #  #          #       #       #     # #     # #     # 
#######  #   #####     #       #       #     # ######  ######  
#     #  #        #    #       #       #     # #   #   #   #   
#     #  #  #     #    #       #     # #     # #    #  #    #  
#     # ###  #####     #        #####  ####### #     # #     # 

def hist_corr_areas_labeling(sessions,corr_type='trace_corr',filternear=True,minNcells=10, 
                        areapairs=' ',layerpairs=' ',projpairs=' ',noise_thr=100,valuematching=None,
                        zscore=False,binres=0.01):
    # areas               = ['V1','PM']
    # redcells            = [0,1]
    # redcelllabels       = ['unl','lab']
    # legendlabels        = np.empty((4,4),dtype='object')

    binedges            = np.arange(-1,1,binres)
    bincenters          = binedges[:-1] + binres/2
    nbins               = len(bincenters)

    if zscore:
        binedges            = np.arange(-5,5,binres)
        bincenters          = binedges[:-1] + binres/2
        nbins               = len(bincenters)

    histcorr           = np.full((nbins,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    meancorr           = np.full((len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    varcorr            = np.full((len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)
    fraccorr           = np.full((2,len(sessions),len(areapairs),len(layerpairs),len(projpairs)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            if valuematching is not None:
                #Get value to match from celldata:
                values  = sessions[ises].celldata[valuematching].to_numpy()

                #For both areas match the values between labeled and unlabeled cells
                idx_V1      = sessions[ises].celldata['roi_name']=='V1'
                idx_PM      = sessions[ises].celldata['roi_name']=='PM'
                group       = sessions[ises].celldata['redcell'].to_numpy()
                idx_sub_V1  = value_matching(np.where(idx_V1)[0],group[idx_V1],values[idx_V1],bins=20,showFig=False)
                idx_sub_PM  = value_matching(np.where(idx_PM)[0],group[idx_PM],values[idx_PM],bins=20,showFig=False)
                
                # matchfilter2d  = np.isin(sessions[ises].celldata.index[:,None], np.concatenate([idx_sub_V1,idx_sub_PM])[None,:])
                # matchfilter    = np.logical_and(matchfilter2d,matchfilter2d.T)

                matchfilter1d = np.zeros(len(sessions[ises].celldata)).astype(bool)
                matchfilter1d[idx_sub_V1] = True
                matchfilter1d[idx_sub_PM] = True

                matchfilter    = np.meshgrid(matchfilter1d,matchfilter1d)
                matchfilter    = np.logical_and(matchfilter[0],matchfilter[1])

            else: 
                matchfilter = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if filternear:
                nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                nearfilter      = np.meshgrid(nearfilter,nearfilter)
                nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
            else: 
                nearfilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            if zscore:
                corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)
            
            rf_type = 'Fsmooth'
            if 'rf_r2_' + rf_type in sessions[ises].celldata:
                el              = sessions[ises].celldata['rf_el_' + rf_type].to_numpy()
                az              = sessions[ises].celldata['rf_az_' + rf_type].to_numpy()
                
                delta_el        = el[:,None] - el[None,:]
                delta_az        = az[:,None] - az[None,:]

                delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
                rffilter        = delta_rf<50
            else: 
                rffilter      = np.ones((len(sessions[ises].celldata),len(sessions[ises].celldata))).astype(bool)

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(sessions[ises].celldata['noise_level']<noise_thr,sessions[ises].celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = ~np.isnan(corrdata)

                        proxfilter      = ~(sessions[ises].distmat_xy<10)

                        cellfilter      = np.all((signalfilter,areafilter,layerfilter,matchfilter,
                                                projfilter,proxfilter,nanfilter,nearfilter,rffilter),axis=0)

                        if np.sum(np.any(cellfilter,axis=0))>minNcells and np.sum(np.any(cellfilter,axis=1))>minNcells:
                            # if ipp==3:
                                # print(np.sum(cellfilter))
                            data      = corrdata[cellfilter].flatten()

                            histcorr[:,ises,iap,ilp,ipp]    = np.histogram(data,bins=binedges,density=True)[0]
                            meancorr[ises,iap,ilp,ipp]      = np.nanmean(data)
                            varcorr[ises,iap,ilp,ipp]       = np.nanstd(data)

                            if corr_type == 'trace_corr':
                                n = len(sessions[ises].ts_F)
                            elif corr_type in ['noise_corr','sig_corr','resp_corr','corr_shuffle']:
                                n = np.shape(sessions[ises].respmat)[1]

                            sigcorrdata = corrdata.copy()
                            sigcorrdata = filter_corr_p(sigcorrdata,n,p_thr=0.01)
                            fraccorr[0,ises,iap,ilp,ipp]       = np.sum(np.logical_and(cellfilter,sigcorrdata>0)) / np.sum(cellfilter)
                            fraccorr[1,ises,iap,ilp,ipp]       = np.sum(np.logical_and(cellfilter,sigcorrdata<0)) / np.sum(cellfilter)

    return bincenters,histcorr,meancorr,varcorr,fraccorr


def filter_corr_p(r,n,p_thr=0.01):
    """Filter out non-significant correlations in a correlation matrix.
    Parameters
    r : array
        Correlation matrix.
    n : int
        Number of datapoints.
    p_thr : float, optional
        Threshold for significant correlations. Default is 0.01.
    Returns
    r : array
        Correlation matrix with non-significant correlations set to nan.
    """
    t           = np.clip(r * np.sqrt((n-2)/(1-r*r)),a_min=-30,a_max=30)#convert correlation to t-statistic
    p           = ss.t.pdf(t, n-2) #convert to p-value using pdf of t-distribution and deg of freedom
    r[p>p_thr]  = np.nan #set all nonsignificant to nan
    # plt.scatter(r.flatten(),p.flatten())
    return r

def filter_sharednan(x,y):
    """
    Filter out shared nans in x and y.
    """
    isnan = np.logical_or(np.isnan(x),np.isnan(y))
    x = x[~isnan]
    y = y[~isnan]
    return x,y



#     # #######    #    #     #     #####  ####### ######  ######  
##   ## #         # #   ##    #    #     # #     # #     # #     # 
# # # # #        #   #  # #   #    #       #     # #     # #     # 
#  #  # #####   #     # #  #  #    #       #     # ######  ######  
#     # #       ####### #   # #    #       #     # #   #   #   #   
#     # #       #     # #    ##    #     # #     # #    #  #    #  
#     # ####### #     # #     #     #####  ####### #     # #     # 

def mean_corr_areas_labeling(sessions,corr_type='trace_corr',absolute=False,
                             filternear=True,filtersign=None,minNcells=10):
    areas               = ['V1','PM']
    redcells            = [0,1]
    redcelllabels       = ['unl','lab']
    legendlabels        = np.empty((4,4),dtype='object')

    meancorr            = np.full((4,4,len(sessions)),np.nan)
    fraccorr            = np.full((4,4,len(sessions)),np.nan)

    for ises in tqdm(range(len(sessions)),desc='Averaging %s across sessions' % corr_type):
        idx_nearfilter = filter_nearlabeled(sessions[ises],radius=50)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            
            if filtersign == 'neg':
                corrdata[corrdata>0] = np.nan
            
            if filtersign =='pos':
                corrdata[corrdata<0] = np.nan

            if absolute:
                corrdata = np.abs(corrdata)

            for ixArea,xArea in enumerate(areas):
                for iyArea,yArea in enumerate(areas):
                    for ixRed,xRed in enumerate(redcells):
                        for iyRed,yRed in enumerate(redcells):

                                idx_source = sessions[ises].celldata['roi_name']==xArea
                                idx_target = sessions[ises].celldata['roi_name']==yArea

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['redcell']==xRed)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['redcell']==yRed)

                                idx_source = np.logical_and(idx_source,sessions[ises].celldata['noise_level']<20)
                                idx_target = np.logical_and(idx_target,sessions[ises].celldata['noise_level']<20)

                                # if 'rf_p_F' in sessions[ises].celldata:
                                #     idx_source = np.logical_and(idx_source,sessions[ises].celldata['rf_p_F']<0.001)
                                    # idx_target = np.logical_and(idx_target,sessions[ises].celldata['rf_p_F']<0.001)

                                # if 'tuning_var' in sessions[ises].celldata:
                                #     idx_source = np.logical_and(idx_source,sessions[ises].celldata['tuning_var']>0.05)
                                #     idx_target = np.logical_and(idx_target,sessions[ises].celldata['tuning_var']>0.05)

                                if filternear:
                                    idx_source = np.logical_and(idx_source,idx_nearfilter)
                                    idx_target = np.logical_and(idx_target,idx_nearfilter)

                                if np.sum(idx_source)>minNcells and np.sum(idx_target)>minNcells:	
                                    meancorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises]  = np.nanmean(corrdata[np.ix_(idx_source, idx_target)])
                                    fraccorr[ixArea*2 + ixRed,iyArea*2 + iyRed,ises] = (
                                        np.sum(~np.isnan(corrdata[np.ix_(idx_source, idx_target)])) /
                                        corrdata[np.ix_(idx_source, idx_target)].size
                                    )

                                legendlabels[ixArea*2 + ixRed,iyArea*2 + iyRed]  = areas[ixArea] + redcelllabels[ixRed] + '-' + areas[iyArea] + redcelllabels[iyRed]

    # assuming meancorr and legeldlabels are 4x4xnSessions array
    upper_tri_indices           = np.triu_indices(4, k=0)
    meancorr_upper_tri          = meancorr[upper_tri_indices[0], upper_tri_indices[1], :]
    fraccorr_upper_tri          = fraccorr[upper_tri_indices[0], upper_tri_indices[1], :]
    
    # assuming legendlabels is a 4x4 array
    # legendlabels_upper_tri      = legendlabels[np.triu_indices(4, k=0)]
    legendlabels_upper_tri      = legendlabels[upper_tri_indices[0], upper_tri_indices[1]]

    df_mean                     = pd.DataFrame(data=meancorr_upper_tri.T,columns=legendlabels_upper_tri)
    df_frac                     = pd.DataFrame(data=fraccorr_upper_tri.T,columns=legendlabels_upper_tri)

    colorder                    = [0,1,4,7,8,9,2,3,5,6]
    legendlabels_upper_tri      = legendlabels_upper_tri[colorder]
    df_mean                     = df_mean[legendlabels_upper_tri]
    df_frac                     = df_frac[legendlabels_upper_tri]

    return df_mean,df_frac

######  ### #     #    #     # #######    #    #     #           #     # #     # 
#     #  #  ##    #    ##   ## #         # #   ##    #    #####   #   #   #   #  
#     #  #  # #   #    # # # # #        #   #  # #   #    #    #   # #     # #   
######   #  #  #  #    #  #  # #####   #     # #  #  #    #    #    #       #    
#     #  #  #   # #    #     # #       ####### #   # #    #    #   # #      #    
#     #  #  #    ##    #     # #       #     # #    ##    #    #  #   #     #    
######  ### #     #    #     # ####### #     # #     #    #####  #     #    #    

def bin_corr_deltaxy(sessions,method='mean',areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='F',
                    rotate_prefori=False,deltaori=None,noise_thr=100,onlysameplane=False,
                    binresolution=5,tuned_thr=0,absolute=False,normalize=False,dsi_thr=0,
                    filtersign=None,corr_thr=0.05,shufflefield=None):
    """
    Binning pairwise correlations as a function of pairwise delta x and y position.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,binedges

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    sig_thr : float, optional
        significance threshold for including cells in the analysis, by default 0.001
    """

    #Binning parameters 2D:
    binlim          = 600
    binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
    bincenters_2d   = binedges_2d[:-1]+binresolution/2 
    nBins           = len(bincenters_2d)

    bin_2d          = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_2d_count    = np.zeros((nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D distance
    binlim          = 600
    binedges_dist   = np.arange(0,binlim,binresolution)+binresolution/2 
    binsdRF = binedges_dist[:-1]+binresolution/2 
    nBins           = len(binsdRF)

    bin_dist        = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_dist_count  = np.zeros((nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D angle
    polarbinres         = 45
    centerthr           = [15,15,15]
    binedges_angle      = np.deg2rad(np.arange(0-polarbinres/2,360,step=polarbinres))
    bincenters_angle    = binedges_angle[:-1]+np.deg2rad(polarbinres/2)
    npolarbins          = len(bincenters_angle)

    bin_angle_cent      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_cent_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    bin_angle_surr      = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_surr_count = np.zeros((npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):

        celldata        = copy.deepcopy(sessions[ises].celldata)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            if shufflefield == 'RF':
                celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type] = my_shuffle_celldata_joint(celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type],
                                                                celldata['roi_name'])
            elif shufflefield == 'XY':
                celldata['xloc'],celldata['yloc'] = my_shuffle_celldata_joint(celldata['xloc'],celldata['yloc'],
                                                                celldata['roi_name'])
            elif shufflefield == 'corrdata':
                corrdata = my_shuffle(corrdata,method='random',axis=None)
            elif shufflefield is not None:
                celldata = my_shuffle_celldata(celldata,shufflefield,keep_roi_name=True)

            delta_x        = celldata['xloc'].to_numpy()[:,None] - celldata['xloc'].to_numpy()[None,:]
            delta_y        = celldata['yloc'].to_numpy()[:,None] - celldata['yloc'].to_numpy()[None,:]
            delta_xy       = np.sqrt(delta_x**2 + delta_y**2)
            angle_xy       = np.mod(np.arctan2(delta_x,delta_y)-np.pi,np.pi*2)
            angle_xy       = np.mod(angle_xy+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
            
            if absolute == True:
                corrdata = np.abs(corrdata)

            if normalize == True:
                corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)

            if method=='mean':
                if filtersign == 'neg':
                    corrsignfilter              = corrdata < 0
                elif filtersign =='pos':
                    corrsignfilter              = corrdata > 0
                else:
                    corrsignfilter = np.ones((len(celldata),len(celldata))).astype(bool)
            elif method=='frac':
                corrsignfilter = np.ones((len(celldata),len(celldata))).astype(bool)
                if filtersign == 'neg':
                    fracsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                elif filtersign =='pos':
                    fracsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                else:
                    raise ValueError('filtersign must be either pos or neg if metohd==frac is chosen')
            else: 
                raise ValueError('invalid method to apply to bins')

            if onlysameplane:
                planefilter    = np.meshgrid(celldata['plane_idx'],celldata['plane_idx'])
                planefilter    = planefilter[0] == planefilter[1]
            else:
                planefilter    = np.ones((len(celldata),len(celldata))).astype(bool)

            for iap,areapair in enumerate(areapairs):
                for ilp,layerpair in enumerate(layerpairs):
                    for ipp,projpair in enumerate(projpairs):
                        signalfilter    = np.meshgrid(celldata['noise_level']<noise_thr,celldata['noise_level']<noise_thr)
                        signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                        if tuned_thr:
                            tuningfilter    = np.meshgrid(celldata['tuning_var']>tuned_thr,celldata['tuning_var']>tuned_thr)
                            tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                        else: 
                            tuningfilter    = np.ones(np.shape(signalfilter))

                        areafilter      = filter_2d_areapair(sessions[ises],areapair)

                        layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                        projfilter      = filter_2d_projpair(sessions[ises],projpair)

                        nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_xy)),axis=0)

                        if deltaori is not None:
                            if isinstance(deltaori,(float,int)):
                                deltaori = np.array([deltaori,deltaori])
                            if np.shape(deltaori) == (1,):
                                deltaori = np.tile(deltaori,2)
                            assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                            delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                            delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                            deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                                    delta_pref <= deltaori[1]),axis=0)
                        else:
                            deltaorifilter = np.ones(np.shape(signalfilter)).astype(bool)

                        #Combine all filters into a single filter:
                        cellfilter      = np.all((signalfilter,tuningfilter,areafilter,corrsignfilter,
                                            layerfilter,projfilter,nanfilter,deltaorifilter),axis=0)

                        if np.any(cellfilter):
                            # valuedata are the correlation values, these are going to be binned
                            vdata           = corrdata[cellfilter].flatten()

                            #First 2D binning: x is elevation, y is azimuth, 
                            xdata               = delta_x[cellfilter].flatten()
                            ydata               = delta_y[cellfilter].flatten()
                            
                            #Take the sum of the correlations in each bin:
                            if method == 'mean': 
                                bin_2d[:,:,iap,ilp,ipp]   += binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='sum')[0]
                            elif method == 'frac':
                                bin_2d[:,:,iap,ilp,ipp]   += np.histogram2d(x=delta_x[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), 
                                        y=delta_y[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), bins=binedges_2d)[0]                                       

                            # Count how many correlation observations are in each bin:
                            bin_2d_count[:,:,iap,ilp,ipp]  += np.histogram2d(x=xdata,y=ydata,bins=binedges_2d)[0]

                            #Now 1D, so only by deltarf:
                            xdata           = delta_xy[cellfilter].flatten()
                            if method == 'mean': 
                                bin_dist[:,iap,ilp,ipp] += binned_statistic(x=xdata,values=vdata,statistic='sum', bins=binedges_dist)[0]
                            elif method == 'frac':
                                bin_dist[:,iap,ilp,ipp] += np.histogram(delta_xy[np.all((cellfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_dist)[0]
                            bin_dist_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_dist)[0]

                            #Now polar binning:
                            tempfilter      = np.all((cellfilter,delta_xy<centerthr[iap]),axis=0)
                            vdata           = corrdata[tempfilter].flatten()
                            xdata           = angle_xy[tempfilter].flatten() #x is angle of rf difference

                            if method == 'mean': 
                                if np.any(tempfilter):
                                    bin_angle_cent[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                statistic='sum',bins=binedges_angle)[0]
                            elif method == 'frac':
                                bin_angle_cent[:,iap,ilp,ipp] += np.histogram(angle_xy[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                            bin_angle_cent_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
                            
                            tempfilter      = np.all((cellfilter,delta_xy>centerthr[iap]),axis=0)
                            vdata           = corrdata[tempfilter].flatten()
                            xdata           = angle_xy[tempfilter].flatten() #x is angle of rf difference
                            
                            if method == 'mean': 
                                if np.any(tempfilter):
                                    bin_angle_surr[:,iap,ilp,ipp]  += binned_statistic(x=xdata,values=vdata,
                                                                statistic='sum',bins=binedges_angle)[0]
                            elif method == 'frac':
                                bin_angle_surr[:,iap,ilp,ipp] += np.histogram(angle_xy[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                            bin_angle_surr_count[:,iap,ilp,ipp] += np.histogram(xdata,bins=binedges_angle)[0]
        
    # divide the total summed correlations by the number of counts in that bin to get the mean:
    bin_2d = bin_2d / bin_2d_count
    bin_dist = bin_dist / bin_dist_count
    bin_angle_cent = bin_angle_cent / bin_angle_cent_count
    bin_angle_surr = bin_angle_surr / bin_angle_surr_count

    return bincenters_2d,bin_2d,bin_2d_count,bin_dist,bin_dist_count,binsdRF,bin_angle_cent,bin_angle_cent_count,bin_angle_surr,bin_angle_surr_count,bincenters_angle


def bin_corr_distance(sessions,areapairs,corr_type='trace_corr',normalize=False,absolute=False):
    binedges = np.arange(0,1000,20) 
    nbins= len(binedges)-1
    binmean = np.full((len(sessions),len(areapairs),nbins),np.nan)
    for ises in tqdm(range(len(sessions)),desc= 'Computing pairwise correlations across antom. distance: '):
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()
            
            if absolute:
                corrdata = np.abs(corrdata)
            # corrdata[corrdata<0] = np.nan
            for iap,areapair in enumerate(areapairs):
                areafilter      = filter_2d_areapair(sessions[ises],areapair)
                nanfilter       = ~np.isnan(corrdata)
                cellfilter      = np.all((areafilter,nanfilter),axis=0)
                # binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xy[cellfilter].flatten(),
                binmean[ises,iap,:] = binned_statistic(x=sessions[ises].distmat_xyz[cellfilter].flatten(),
                                                    values=corrdata[cellfilter].flatten(),
                                                    statistic='mean', bins=binedges)[0]
            
    if normalize: # subtract mean NC from every session:
        binmean = binmean - np.nanmean(binmean[:,:,binedges[:-1]<600],axis=2,keepdims=True)

    return binmean,binedges


def plot_bin_corr_distance(sessions,binmean,binedges,areapairs,corr_type):
    clrs_areapairs = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs = [clrs_areapairs]
    fig,axes = plt.subplots(1,1,figsize=(3.5,3))
    handles = []
    ax = axes
    for iap,areapair in enumerate(areapairs):
        for ises in range(len(sessions)):
            ax.plot(binedges[:-1],binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
        handles.append(shaded_error(ax=ax,x=binedges[:-1],y=binmean[:,iap,:].squeeze(),
                                    error='sem',color=clrs_areapairs[iap],linewidth=3))
        # plt.savefig(os.path.join(figdir,'NoiseCorr_distRF_RegressOut_' + areapair + '_' + sessions[sesidx].sessiondata['session_id'][0] + '.png'), format = 'png')

    ax.legend(handles,areapairs,loc='upper right',frameon=False,fontsize=9)	
    ax.set_xlabel('Anatomical distance ($\mu$m)')
    ax.set_ylabel('Correlation')
    ax.set_xlim([20,600])
    ax_nticks(ax,3)
    # ax.set_title('%s (%s)' % (corr_type,protocol))
    # ax.set_ylim([-0.01,0.04])
    # ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_ylim([0,0.04])
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=8)
    sns.despine(top=True,right=True,offset=3)
    plt.tight_layout()
    return fig


def plot_bin_corr_distance_projs(binsdRF,bin_dist,areapairs,layerpairs,projpairs):
    clrs_projpairs = get_clr_labelpairs(projpairs)
    clrs_areapairs = get_clr_area_pairs(areapairs)
    # nSessions = binsdRF.shape[0]
    nprojpairs = len(projpairs)
    nareapairs = len(areapairs)

    ilp = 0
    fig,axes = plt.subplots(1,nareapairs,figsize=(6.5,3),sharey=True,sharex=True)
    handles = []
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        for ipp,projpair in enumerate(projpairs):
            ax.plot(binsdRF,bin_dist[:,iap,ilp,ipp].squeeze(),
                                        color=clrs_projpairs[ipp],linewidth=3)
            # handles.append(shaded_error(x=binsdRF,y=bin_dist[:,iap,ilp,ipp].squeeze(),ax=ax,
                                        # error='sem',color=clrs_projpairs[ipp],linewidth=3))
        # data = 
        # for ises in range(nSessions):
            # ax.plot(binsdRF,binmean[ises,iap,:].squeeze(),linewidth=0.15,color=clrs_areapairs[iap])
        # handles.append(shaded_error(ax=ax,x=binsdRF,y=bin_dist[:,iap,ilp,ipp].squeeze(),
                                    # error='sem',color=clrs_areapairs[iap],linewidth=3))

        ax.legend(projpairs,loc='upper right',frameon=False,fontsize=9)	
        ax.set_xlabel('Anatomical distance ($\mu$m)')
        ax.set_ylabel('Correlation')
        ax.set_xlim([20,600])
        ax_nticks(ax,3)
    # ax.set_title('%s (%s)' % (corr_type,protocol))
    # ax.set_ylim([-0.01,0.04])
    # ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_ylim([0,0.04])
        ax.tick_params(axis='both', which='major', labelsize=8)
    sns.despine(top=True,right=True,offset=3)
    plt.tight_layout()
    return fig

######  ### #     #    ######  ####### #       #######    #       ######  ####### 
#     #  #  ##    #    #     # #       #          #      # #      #     # #       
#     #  #  # #   #    #     # #       #          #     #   #     #     # #       
######   #  #  #  #    #     # #####   #          #    #     #    ######  #####   
#     #  #  #   # #    #     # #       #          #    #######    #   #   #       
#     #  #  #    ##    #     # #       #          #    #     #    #    #  #       
######  ### #     #    ######  ####### #######    #    #     #    #     # #       


def bin_corr_deltarf_ses(sessions,method='mean',areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='Fsmooth',
                    r2_thr=0.2,noise_thr=100,filternear=False,binresolution=5,binlim=75,tuned_thr=0,absolute=False,
                    normalize=False,dsi_thr=0,min_dist=15,filtersign=None,corr_thr=0.05,
                    rotate_prefori=False,deltaori=None,centerori=None,surroundori=None,shufflefield=None):
    """
    Binning pairwise correlations as a function of pairwise delta azimuth and elevation.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,binedges

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    rf_type : str, optional
        type of receptive field to use, by default 'F'
    """
    nSessions = len(sessions)

    #Binning parameters 2D:
    binedges_2d     = np.arange(-binlim,binlim,binresolution)+binresolution/2 
    bincenters_2d   = binedges_2d[:-1]+binresolution/2 
    nBins           = len(bincenters_2d)

    bin_2d          = np.zeros((nSessions,nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_2d_count    = np.zeros((nSessions,nBins,nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D distance
    binedges_dist   = np.arange(-binresolution/2,binlim,binresolution)+binresolution/2 
    binsdRF = binedges_dist[:-1]+binresolution/2 
    nBins           = len(binsdRF)

    bin_dist        = np.zeros((nSessions,nBins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_dist_count  = np.zeros((nSessions,nBins,len(areapairs),len(layerpairs),len(projpairs)))

    #Binning parameters 1D angle
    polarbinres         = 90
    binedges_angle      = np.deg2rad(np.arange(0-polarbinres/2,360,step=polarbinres))
    bincenters_angle    = binedges_angle[:-1]+np.deg2rad(polarbinres/2)
    npolarbins          = len(bincenters_angle)

    # centerthr           = [15,15,15,15]
    centerthr           = [20,20,20,20]
    bin_angle_cent      = np.zeros((nSessions,npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_cent_count = np.zeros((nSessions,npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    bin_angle_surr      = np.zeros((nSessions,npolarbins,len(areapairs),len(layerpairs),len(projpairs)))
    bin_angle_surr_count = np.zeros((nSessions,npolarbins,len(areapairs),len(layerpairs),len(projpairs)))

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
        celldata = copy.deepcopy(sessions[ises].celldata)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            if shufflefield == 'RF':
                celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type] = my_shuffle_celldata_joint(celldata['rf_el_' + rf_type],
                                                                celldata['rf_az_' + rf_type],celldata['roi_name'])
            elif shufflefield == 'XY':
                celldata['xloc'],celldata['yloc'] = my_shuffle_celldata_joint(celldata['xloc'],celldata['yloc'],
                                                                celldata['roi_name'])
            elif shufflefield == 'corrdata':
                corrdata = my_shuffle(corrdata,method='random',axis=None)
            elif shufflefield is not None:
                celldata = my_shuffle_celldata(celldata,shufflefield,keep_roi_name=True)

            if 'rf_r2_' + rf_type in celldata:

                el              = celldata['rf_el_' + rf_type].to_numpy()
                az              = celldata['rf_az_' + rf_type].to_numpy()
                
                delta_el        = el[:,None] - el[None,:]
                delta_az        = az[:,None] - az[None,:]

                delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
                angle_rf        = np.mod(np.arctan2(delta_az,delta_el)+np.pi/2,np.pi*2)
                angle_rf        = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)

                # x = delta_az.flatten() #control plot that azimuth and elevation are jointly mapped onto correct angles:
                # y = delta_el.flatten()
                # c = angle_rf.flatten() / np.pi
                # c = delta_rf.flatten() / np.pi
                # plt.scatter(x[:1000],y[:1000],c=c[:1000])

                # Careful definitions:
                # delta_az is source neurons azimuth minus target neurons azimuth position:
                # plt.imshow(delta_az[:10,:10],vmin=-20,vmax=20,cmap='bwr')
                # entry delta_az[0,1] being positive means target neuron RF is to the right of source neuron
                # entry delta_el[0,1] being positive means target neuron RF is above source neuron
                # To rotate azimuth and elevation to relative to the preferred orientation of the source neuron
                # means that for a neuron with preferred orientation 45 deg all delta az and delta el of paired neruons
                # will rotate 45 deg, such that now delta azimuth and delta elevation is relative to the angle 
                # of pref ori of the source neuron

                if absolute:
                    corrdata = np.abs(corrdata)

                if normalize:
                    # corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)
                    corrdata = corrdata - np.nanmean(corrdata,axis=None)

                if sessions[ises].sessiondata['protocol'][0] == 'SP':
                    n = len(sessions[ises].ts_F)
                elif corr_type == 'trace_corr':
                    n = len(sessions[ises].ts_F)
                elif corr_type in ['noise_corr','noise_cov','sig_corr']:
                    n = np.shape(sessions[ises].respmat)[1]
                sigcorrdata = corrdata.copy()

                if method=='mean':
                    if filtersign == 'neg':
                        # corrsignfilter              = corrdata < -0.1
                        # corrsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                        corrsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) < 0
                    elif filtersign =='pos':
                        # corrsignfilter              = corrdata > 0.3
                        # corrsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                        corrsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) > 0
                    else:
                        corrsignfilter = np.ones((len(celldata),len(celldata))).astype(bool)
                elif method=='frac':
                    corrsignfilter = np.ones((len(celldata),len(celldata))).astype(bool)
                    if filtersign == 'neg':
                        # fracsignfilter              = corrdata < np.nanpercentile(corrdata,(corr_thr*100))
                        # fracsignfilter              = corrdata < -0.15
                        fracsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) < 0
                    elif filtersign =='pos':
                        # fracsignfilter              = corrdata > np.nanpercentile(corrdata,(100-corr_thr*100))
                        # fracsignfilter              = corrdata > 0.3
                        fracsignfilter              = filter_corr_p(sigcorrdata,n,p_thr=corr_thr) > 0
                    else:
                        raise ValueError('filtersign must be either pos or neg if metohd==frac is chosen')
                else: 
                    raise ValueError('invalid method to apply to bins')

                if filternear:
                    nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                    nearfilter      = np.meshgrid(nearfilter,nearfilter)
                    nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
                else: 
                    nearfilter      = np.ones((len(celldata),len(celldata))).astype(bool)

                # Rotate delta azimuth and delta elevation to the pref ori of the source neuron
                # delta_az is source neurons
                if rotate_prefori: 
                    for iN in range(len(celldata)):
                        # ori_rots            = celldata['pref_ori'][iN]
                        ori_rots            = 360 - np.tile(celldata['pref_ori'][iN],len(celldata))
                        angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
                        angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots) 
                        delta_el[iN,:]      = angle_vec_rot[0,:]
                        delta_az[iN,:]      = angle_vec_rot[1,:]

                    delta_rf         = np.sqrt(delta_az**2 + delta_el**2)
                    angle_rf         = np.mod(np.arctan2(delta_az,delta_el)+np.pi/2,np.pi*2)
                    angle_rf         = np.mod(angle_rf+np.deg2rad(polarbinres/2),np.pi*2) - np.deg2rad(polarbinres/2)
                    # plt.hist(angle_rf.flatten())

                # plt.scatter(angle_rf_b[celldata['pref_ori']==90,:].flatten(),angle_rf[celldata['pref_ori']==90,:].flatten())

                rffilter        = np.meshgrid(celldata['rf_r2_' + rf_type]> r2_thr,celldata['rf_r2_'  + rf_type] > r2_thr)
                rffilter        = np.logical_and(rffilter[0],rffilter[1])
                
                signalfilter    = np.meshgrid(celldata['noise_level']<noise_thr,celldata['noise_level']<noise_thr)
                signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                if tuned_thr:
                    if tuned_thr<1:
                        tuningfilter    = np.meshgrid(celldata['tuning_var']>tuned_thr,celldata['tuning_var']>tuned_thr)
                    elif tuned_thr>1:
                        tuningfilter    = np.meshgrid(celldata['gOSI']>np.percentile(celldata['gOSI'],100-tuned_thr),
                                                      celldata['gOSI']>np.percentile(celldata['gOSI'],100-tuned_thr))
                    tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                else: 
                    tuningfilter    = np.ones(np.shape(rffilter))

                nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_rf)),axis=0)

                proxfilter      = ~(sessions[ises].distmat_xy<min_dist)

                # assert sum([deltaori is not None, centerori is not None, surroundori is not None]) <= 1, 'at maximum one of deltaori, centerori, or surroundori can be not None'
                
                if centerori is not None:
                    centerorifilter = np.tile(celldata['pref_ori']== centerori,(len(celldata),1)).T
                else:
                    centerorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if surroundori is not None:
                    surroundorifilter = np.tile(celldata['pref_ori']== surroundori,(len(celldata),1))
                else:
                    surroundorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if deltaori is not None:
                    if isinstance(deltaori,(float,int)):
                        deltaori = np.array([deltaori,deltaori])
                    if np.shape(deltaori) == (1,):
                        deltaori = np.tile(deltaori,2)
                    assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                    delta_pref = sessions[ises].delta_pref.copy()
                    # delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                    # delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                    deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                            delta_pref <= deltaori[1]),axis=0)
                else:
                    deltaorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if dsi_thr:
                    dsi_filter = np.meshgrid(celldata['DSI']>dsi_thr,celldata['DSI']>dsi_thr)
                    dsi_filter = np.logical_and(dsi_filter[0],dsi_filter[1])
                else:
                    dsi_filter = np.ones(np.shape(rffilter)).astype(bool)

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)
                            #Combine all filters into a single filter:
                            cellfilter      = np.all((rffilter,signalfilter,tuningfilter,areafilter,nearfilter,corrsignfilter,
                                                layerfilter,projfilter,proxfilter,nanfilter,
                                                deltaorifilter,dsi_filter,centerorifilter,surroundorifilter),axis=0)
                            minNcells = 10

                            if np.any(cellfilter) and np.sum(np.any(cellfilter,axis=0)) > minNcells and np.sum(np.any(cellfilter,axis=1)) > minNcells:
                                # valuedata are the correlation values, these are going to be binned
                                vdata               = corrdata[cellfilter].flatten()

                                #First 2D binning: x is elevation, y is azimuth, 
                                xdata               = delta_el[cellfilter].flatten()
                                ydata               = delta_az[cellfilter].flatten()
                                #First 2D binning: x is azimuth, y is elevation, 
                                # xdata               = delta_az[cellfilter].flatten()
                                # ydata               = delta_el[cellfilter].flatten()
                                
                                #Take the sum of the correlations in each bin:
                                if method == 'mean': 
                                    bin_2d[ises,:,:,iap,ilp,ipp]   = binned_statistic_2d(x=xdata, y=ydata, values=vdata,bins=binedges_2d, statistic='sum')[0]
                                elif method == 'frac':
                                    bin_2d[ises,:,:,iap,ilp,ipp]   = np.histogram2d(x=delta_az[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), 
                                            y=delta_el[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), bins=binedges_2d)[0]                                       
                                    # bin_2d[:,:,iap,ilp,ipp]   += np.histogram2d(x=delta_el[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), 
                                            # y=delta_az[np.all((cellfilter,fracsignfilter),axis=0)].flatten(), bins=binedges_2d)[0]                                       

                                # Count how many correlation observations are in each bin:
                                bin_2d_count[ises,:,:,iap,ilp,ipp]  = np.histogram2d(x=xdata,y=ydata,bins=binedges_2d)[0]

                                #Now 1D, so only by deltarf:
                                xdata           = delta_rf[cellfilter].flatten()
                                if method == 'mean': 
                                    bin_dist[ises,:,iap,ilp,ipp] = binned_statistic(x=xdata,values=vdata,statistic='sum', bins=binedges_dist)[0]
                                elif method == 'frac':
                                    bin_dist[ises,:,iap,ilp,ipp] = np.histogram(delta_rf[np.all((cellfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_dist)[0]
                                bin_dist_count[ises,:,iap,ilp,ipp] = np.histogram(xdata,bins=binedges_dist)[0]

                                #Now polar binning:
                                tempfilter      = np.all((cellfilter,delta_rf<centerthr[iap]),axis=0)
                                vdata           = corrdata[tempfilter].flatten()
                                xdata           = angle_rf[tempfilter].flatten() #x is angle of rf difference

                                if method == 'mean': 
                                    if np.any(tempfilter):
                                        bin_angle_cent[ises,:,iap,ilp,ipp]  = binned_statistic(x=xdata,values=vdata,
                                                                    statistic='sum',bins=binedges_angle)[0]
                                elif method == 'frac':
                                    bin_angle_cent[ises,:,iap,ilp,ipp] = np.histogram(angle_rf[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                                bin_angle_cent_count[ises,:,iap,ilp,ipp] = np.histogram(xdata,bins=binedges_angle)[0]
                                
                                tempfilter      = np.all((cellfilter,delta_rf>centerthr[iap]),axis=0)
                                vdata           = corrdata[tempfilter].flatten()
                                xdata           = angle_rf[tempfilter].flatten() #x is angle of rf difference
                                
                                if method == 'mean': 
                                    if np.any(tempfilter):
                                        bin_angle_surr[ises,:,iap,ilp,ipp]  = binned_statistic(x=xdata,values=vdata,
                                                                    statistic='sum',bins=binedges_angle)[0]
                                elif method == 'frac':
                                    bin_angle_surr[ises,:,iap,ilp,ipp] = np.histogram(angle_rf[np.all((tempfilter,fracsignfilter),axis=0)].flatten(),bins=binedges_angle)[0]
                                bin_angle_surr_count[ises,:,iap,ilp,ipp] = np.histogram(xdata,bins=binedges_angle)[0]
        

    # divide the total summed correlations by the number of counts in that bin to get the mean:
    with np.errstate(invalid='ignore'):
        bin_2d = bin_2d / bin_2d_count
        bin_dist = bin_dist / bin_dist_count
        bin_angle_cent = bin_angle_cent / bin_angle_cent_count
        bin_angle_surr = bin_angle_surr / bin_angle_surr_count

    return bincenters_2d,bin_2d,bin_2d_count,bin_dist,bin_dist_count,binsdRF,bin_angle_cent,bin_angle_cent_count,bin_angle_surr,bin_angle_surr_count,bincenters_angle


def bin_corr_deltarf_ses_vkeep(sessions,method='mean',areapairs=' ',layerpairs=' ',projpairs=' ',corr_type='noise_corr',rf_type='Fsmooth',
                    r2_thr=0.2,noise_thr=100,filternear=False,binresolution=5,tuned_thr=0,absolute=False,
                    normalize=False,dsi_thr=0,min_dist=15,filtersign=None,corr_thr=0.05,
                    rotate_prefori=False,deltaori=None,centerori=None,surroundori=None,shufflefield=None):
    """
    Binning pairwise correlations as a function of pairwise delta azimuth and elevation.
    - Sessions are binned by areapairs, layerpairs, and projpairs.
    - Returns binmean,bincount,binedges

    Parameters
    ----------
    sessions : list
        list of sessions
    areapairs : list (if ' ' then all areapairs are used)
        list of areapairs
    layerpairs : list  (if ' ' then all layerpairs are used)
        list of layerpairs
    projpairs : list  (if ' ' then all projpairs are used)
        list of projpairs
    corr_type : str, optional
        type of correlation to use, by default 'trace_corr'
    normalize : bool, optional
        whether to normalize correlations to the mean correlation at distances < 60 um, by default False
    rf_type : str, optional
        type of receptive field to use, by default 'F'
    """
    nSessions = len(sessions)
    maxsamples = 10000

    #Binning parameters 1D distance
    binlim          = 75
    binedges_dist   = np.arange(-binresolution/2,binlim,binresolution)+binresolution/2 
    binsdRF = binedges_dist[:-1]+binresolution/2 
    nBins           = len(binsdRF)

    bin_dist        = np.full((nSessions,nBins,len(areapairs),len(layerpairs),len(projpairs),maxsamples),np.nan)
    bin_dist_count  = np.full((nSessions,nBins,len(areapairs),len(layerpairs),len(projpairs),maxsamples),0)

    for ises in tqdm(range(len(sessions)),total=len(sessions),desc= 'Computing 2D corr histograms maps: '):
        celldata = copy.deepcopy(sessions[ises].celldata)
        if hasattr(sessions[ises],corr_type):
            corrdata = getattr(sessions[ises],corr_type).copy()

            if shufflefield == 'RF':
                celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type] = my_shuffle_celldata_joint(celldata['rf_el_' + rf_type],celldata['rf_az_' + rf_type],
                                                                celldata['roi_name'])
            elif shufflefield == 'XY':
                celldata['xloc'],celldata['yloc'] = my_shuffle_celldata_joint(celldata['xloc'],celldata['yloc'],
                                                                celldata['roi_name'])
            elif shufflefield == 'corrdata':
                corrdata = my_shuffle(corrdata,method='random',axis=None)
            elif shufflefield is not None:
                celldata = my_shuffle_celldata(celldata,shufflefield,keep_roi_name=True)

            if 'rf_r2_' + rf_type in celldata:

                el              = celldata['rf_el_' + rf_type].to_numpy()
                az              = celldata['rf_az_' + rf_type].to_numpy()
                
                delta_el        = el[:,None] - el[None,:]
                delta_az        = az[:,None] - az[None,:]

                delta_rf        = np.sqrt(delta_az**2 + delta_el**2)
                
                # Careful definitions:
                # delta_az is source neurons azimuth minus target neurons azimuth position:
                # plt.imshow(delta_az[:10,:10],vmin=-20,vmax=20,cmap='bwr')
                # entry delta_az[0,1] being positive means target neuron RF is to the right of source neuron
                # entry delta_el[0,1] being positive means target neuron RF is above source neuron
                # To rotate azimuth and elevation to relative to the preferred orientation of the source neuron
                # means that for a neuron with preferred orientation 45 deg all delta az and delta el of paired neruons
                # will rotate 45 deg, such that now delta azimuth and delta elevation is relative to the angle 
                # of pref ori of the source neuron 

                if absolute:
                    corrdata = np.abs(corrdata)

                if normalize:
                    # corrdata = corrdata/np.nanstd(corrdata,axis=None) - np.nanmean(corrdata,axis=None)
                    corrdata = corrdata - np.nanmean(corrdata,axis=None)

                if filternear:
                    nearfilter      = filter_nearlabeled(sessions[ises],radius=50)
                    nearfilter      = np.meshgrid(nearfilter,nearfilter)
                    nearfilter      = np.logical_and(nearfilter[0],nearfilter[1])
                else: 
                    nearfilter      = np.ones((len(celldata),len(celldata))).astype(bool)

                # Rotate delta azimuth and delta elevation to the pref ori of the source neuron
                # delta_az is source neurons
                if rotate_prefori: 
                    for iN in range(len(celldata)):
                        # ori_rots            = celldata['pref_ori'][iN]
                        ori_rots            = 360 - np.tile(celldata['pref_ori'][iN],len(celldata))
                        angle_vec           = np.vstack((delta_el[iN,:], delta_az[iN,:]))
                        angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots) 
                        # angle_vec_rot       = apply_ori_rot(angle_vec,ori_rots + 90) #90 degrees is added to make collinear horizontal, incorrect
                        delta_el[iN,:]      = angle_vec_rot[0,:]
                        delta_az[iN,:]      = angle_vec_rot[1,:]

                    delta_rf         = np.sqrt(delta_az**2 + delta_el**2)

                rffilter        = np.meshgrid(celldata['rf_r2_' + rf_type]> r2_thr,celldata['rf_r2_'  + rf_type] > r2_thr)
                rffilter        = np.logical_and(rffilter[0],rffilter[1])
                
                signalfilter    = np.meshgrid(celldata['noise_level']<noise_thr,celldata['noise_level']<noise_thr)
                signalfilter    = np.logical_and(signalfilter[0],signalfilter[1])

                if tuned_thr:
                    tuningfilter    = np.meshgrid(celldata['tuning_var']>tuned_thr,celldata['tuning_var']>tuned_thr)
                    tuningfilter    = np.logical_and(tuningfilter[0],tuningfilter[1])
                else: 
                    tuningfilter    = np.ones(np.shape(rffilter))

                nanfilter       = np.all((~np.isnan(corrdata),~np.isnan(delta_rf)),axis=0)

                proxfilter      = ~(sessions[ises].distmat_xy<min_dist)

                assert sum([deltaori is not None, centerori is not None, surroundori is not None]) <= 1, 'at maximum one of deltaori, centerori, or surroundori can be not None'
                
                if centerori is not None:
                    centerorifilter = np.tile(celldata['pref_ori']== centerori,(len(celldata),1)).T
                else:
                    centerorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if surroundori is not None:
                    surroundorifilter = np.tile(celldata['pref_ori']== surroundori,(len(celldata),1))
                else:
                    surroundorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if deltaori is not None:
                    if isinstance(deltaori,(float,int)):
                        deltaori = np.array([deltaori,deltaori])
                    if np.shape(deltaori) == (1,):
                        deltaori = np.tile(deltaori,2)
                    assert np.shape(deltaori) == (2,),'deltaori must be a 2x1 array'
                    delta_pref = sessions[ises].delta_pref.copy()
                    # delta_pref = np.mod(sessions[ises].delta_pref,90) #convert to 0-90, direction tuning is ignored
                    # delta_pref[sessions[ises].delta_pref == 90] = 90 #after modulo operation, restore 90 as 90
                    deltaorifilter = np.all((delta_pref >= deltaori[0], #find all entries with delta_pref between deltaori[0] and deltaori[1]
                                            delta_pref <= deltaori[1]),axis=0)
                else:
                    deltaorifilter = np.ones(np.shape(rffilter)).astype(bool)

                if dsi_thr:
                    dsi_filter = np.meshgrid(celldata['DSI']>dsi_thr,celldata['DSI']>dsi_thr)
                    dsi_filter = np.logical_and(dsi_filter[0],dsi_filter[1])
                else:
                    dsi_filter = np.ones(np.shape(rffilter)).astype(bool)

                for iap,areapair in enumerate(areapairs):
                    for ilp,layerpair in enumerate(layerpairs):
                        for ipp,projpair in enumerate(projpairs):

                            areafilter      = filter_2d_areapair(sessions[ises],areapair)

                            layerfilter     = filter_2d_layerpair(sessions[ises],layerpair)

                            projfilter      = filter_2d_projpair(sessions[ises],projpair)

                            #Combine all filters into a single filter:
                            cellfilter      = np.all((rffilter,signalfilter,tuningfilter,areafilter,nearfilter,
                                                layerfilter,projfilter,proxfilter,nanfilter,
                                                deltaorifilter,dsi_filter,centerorifilter,surroundorifilter),axis=0)

                            if np.any(cellfilter):
                                # valuedata are the correlation values, these are going to be binned
                                vdata               = corrdata[cellfilter].flatten()
                                #1D binning by deltarf:
                                xdata           = delta_rf[cellfilter].flatten()
                                
                                for ibin in range(len(binedges_dist)-1):
                                    idx  = (xdata >= binedges_dist[ibin]) & (xdata < binedges_dist[ibin+1])
                                    bin_dist[ises,ibin,iap,ilp,ipp,:np.sum(idx)] = vdata[idx][:maxsamples]
                                bin_dist_count[ises,:,iap,ilp,ipp] = np.histogram(xdata,bins=binedges_dist)[0]
    
    return bin_dist,bin_dist_count,binsdRF



######  #       ####### #######    ######  ####### #       #######    #       ######  ####### 
#     # #       #     #    #       #     # #       #          #      # #      #     # #       
#     # #       #     #    #       #     # #       #          #     #   #     #     # #       
######  #       #     #    #       #     # #####   #          #    #     #    ######  #####   
#       #       #     #    #       #     # #       #          #    #######    #   #   #       
#       #       #     #    #       #     # #       #          #    #     #    #    #  #       
#       ####### #######    #       ######  ####### #######    #    #     #    #     # #       


def plot_corr_radial_tuning_areas_sessions(binsdRF,bin_dist_count_ses,bin_dist_data_ses,	
                           areapairs=' ',layerpairs=' ',projpairs=' ',datatype='Correlation',
                           min_counts=100):
    if np.max(binsdRF)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        # xylim               = 65
        xylim               = 70
        dim12label = 'RF (\N{DEGREE SIGN})'

    #Colors:
    clrs_areapairs      = get_clr_area_pairs(areapairs) 
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    #Compute data mean and error:
    bin_dist_data_ses = copy.deepcopy(bin_dist_data_ses)
    bin_dist_data_ses[bin_dist_count_ses<min_counts] = np.nan
    data_mean   = np.nanmean(bin_dist_data_ses,axis=0)
    data_error  = np.nanstd(bin_dist_data_ses,axis=0) / np.sqrt(np.shape(bin_dist_data_ses)[0])

    fig,axes    = plt.subplots(1,len(areapairs),figsize=(2*len(areapairs),3),sharex=True,sharey=True)
    if len(areapairs)==1: 
        axes = [axes]
    ilp = 0
    ipp = 0
    handles = []

    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        # bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5

        ax.plot(binsdRF,bin_dist_data_ses[:,:,iap,ilp,ipp].T,color=clrs_areapairs[iap],alpha=0.5,linewidth=0.5)
        handles.append(shaded_error(x=binsdRF,y=data_mean[:,iap,ilp,ipp],yerror=data_error[:,iap,ilp,ipp],
                        ax = ax,color=clrs_areapairs[iap],label=areapair))
        bindata = data_mean[:,iap,ilp,ipp]
        xdata = binsdRF[~np.isnan(bindata)]
        ydata = bindata[~np.isnan(bindata)]

        try:
            # slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
            # ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} linfit',linewidth=1)
            
            popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[0]-ydata[-1], 0, ydata[-1]],bounds=(-10, 10))
            # popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[0]-ydata[-1], 0, ydata[-1]])
            ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} fit',linewidth=1)
            print('Spatial constant %s: %1.4f' % (areapair,popt[1]))
            print('Amplitude %s: %0.4f' % (areapair,popt[0]))
            print('Offset %s: %0.4f' % (areapair,popt[2]))
            # print('Spatial constant %s: %2.2f' % (areapair,popt[1]))
        except:
            print('curve_fit failed for %s' % (areapair))
            continue
        
        # ax.legend(handles=handles,labels=areapairs,frameon=False)
        ax.set_xlim([0,xylim])
        if datatype=='Correlation':
            # ax.set_ylim([0.01,0.08])
            # ax.set_ylim([0.01,0.12])
            # ax.set_ylim([my_floor(np.nanmin(bin_dist_data_ses),2),my_ceil(np.nanmax(bin_dist_data_ses),2)])
            ax.set_ylim([my_floor(np.nanpercentile(bin_dist_data_ses,5),2),my_ceil(np.nanpercentile(bin_dist_data_ses,98),2)])
        else:
            # ax.set_ylim([my_floor(np.nanmin(bin_dist_data_ses),2),my_ceil(np.nanmax(bin_dist_data_ses),2)])
            ax.set_ylim([my_floor(np.nanpercentile(bin_dist_data_ses,5),2),my_ceil(np.nanpercentile(bin_dist_data_ses,98),2)])
        
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
        if iap==0:
            ax.set_ylabel(datatype)
    sns.despine(fig=fig,top=True,right=True,offset=5)
    plt.tight_layout()
    return fig

def plot_corr_radial_tuning_areas(binsdRF,bin_dist_count_ses,bin_dist_data_ses,	
                           areapairs=' ',layerpairs=' ',projpairs=' ',datatype='Correlation'):
    if np.max(binsdRF)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 65
        dim12label = 'RF (\N{DEGREE SIGN})'

    min_counts      = 100

    #Colors:
    clrs_areapairs      = get_clr_area_pairs(areapairs) 
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    # bin_dist_data_ses -= np.nanmean(bin_dist_data_ses,axis=1,keepdims=True)
    #Compute data mean and error:
    bin_dist_data_ses[bin_dist_count_ses<min_counts] = np.nan
    data_mean   = np.nanmean(bin_dist_data_ses,axis=0)
    data_error  = np.nanstd(bin_dist_data_ses,axis=0) / np.sqrt(np.shape(bin_dist_data_ses)[0])

    # Number of bootstrap iterations
    ilp = 0
    ipp = 0
    n_bootstrap     = 500
    paramdata       = np.full((3, len(areapairs), n_bootstrap), np.nan)
    paramlabels     = ['amplitude','decay','offset']
    for iap,areapair in enumerate(areapairs):
        xdata = binsdRF
        nses = np.shape(bin_dist_data_ses)[0]
        for iboot in range(n_bootstrap):
            try:
                idx         = np.random.choice(nses,nses,replace=True)
                # idx         = np.random.choice(nses,int(nses/2),replace=False)
                bindata     = np.nanmean(bin_dist_data_ses[idx,:,iap,ilp,ipp],axis=0)
                ydata       = bindata[~np.isnan(bindata)]
                # popt, pcov  = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[0]-ydata[-1], 0.05, ydata[-1]],bounds=([-1,0,0], [1,1,0.1]))
                popt, pcov  = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[0]-ydata[-1], 0.05, ydata[-1]],
                                        bounds=((-0.3, 0, 0), (0.3, 1, 1)))
                paramdata[:,iap,iboot] = popt
            except:
                continue

    fig,axes    = plt.subplots(1,4,figsize=(12,3))
    ilp = 0
    ipp = 0
    handles = []
    ax = axes[0]
    for iap,areapair in enumerate(areapairs):
        # bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        handles.append(shaded_error(x=binsdRF,y=data_mean[:,iap,ilp,ipp],yerror=data_error[:,iap,ilp,ipp],
                        ax = ax,color=clrs_areapairs[iap],label=areapair))
        bindata = data_mean[:,iap,ilp,ipp]
        xdata = binsdRF[~np.isnan(bindata)]
        ydata = bindata[~np.isnan(bindata)]
        try:
            # slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
            # ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} linfit',linewidth=1)
            
            popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[0]-ydata[-1], 0.05, ydata[-1]],bounds=(-10, 10))
            ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} fit',linewidth=1)

        except:
            print('curve_fit failed for %s' % (areapair))
            continue
        
    ax.legend(handles=handles,labels=areapairs,frameon=False)
    ax.set_xlim([0,xylim])
    ax.set_ylim([my_floor(np.nanmin(data_mean)*0.65,3),my_ceil(np.nanmax(data_mean)*1.1,3)])
    ax.set_xlabel(u'Δ %s' % dim12label)   
    # ax.set_title('%s\n Joint' % (areapair),c=clrs_areapairs[iap])
    ax.set_ylabel(datatype)

    # ax = axes[1]
    # for iap,areapair in enumerate(areapairs):
    #     data = np.empty((len(binsdRF),n_bootstrap))
    #     for iboot in range(n_bootstrap):
    #         data[:,iboot] = paramdata[0,iap,iboot] * np.exp(-paramdata[1,iap,iboot] * xdata) + paramdata[2,iap,iboot]
        
    #     h, = ax.plot(binsdRF,np.nanpercentile(data,50,axis=1),color=clrs_areapairs[iap],linestyle='--',label=areapair)
    #     ax.fill_between(binsdRF, np.nanpercentile(data,5,axis=1), np.nanpercentile(data,95,axis=1),color=clrs_areapairs[iap],alpha=0.2)

    # ax.legend(handles=handles,labels=areapairs,frameon=False)
    # ax.set_xlim([0,xylim])
    # ax.set_ylim([my_floor(np.nanmin(data_mean)*0.65,3),my_ceil(np.nanmax(data_mean)*1.1,3)])
    # ax.set_xlabel(u'Δ %s' % dim12label)   
    # ax.set_ylabel(datatype)

    for ip in range(3):
        ax = axes[ip+1]

        # sns.boxplot(data=paramdata[ip,:,:].T,ax=ax,whis=[10, 90],palette=clrs_areapairs,showfliers=False)
        sns.boxplot(data=paramdata[ip,:,:].T,ax=ax,whis=1,palette=clrs_areapairs,showfliers=False)
        # sns.violinplot(data=paramdata[ip,:,:].T,ax=ax,palette=clrs_areapairs,showfliers=False)
        # sns.boxplot(paramdata[ip,:,:].T,ax=ax,palette=clrs_areapairs,showfliers=False)
        ax.set_title(paramlabels[ip])
        ax.set_xlim([-0.5,2.5])
        ax.set_xticklabels(areapairs)
        ax.axhline(0,linestyle='--',color='k',linewidth=1)

    plt.tight_layout()
    sns.despine(top=True,right=True,offset=3)
    return fig

def plot_corr_radial_tuning_areas_mean(binsdRF,bin_dist_count,bin_dist_mean,	
                           areapairs=' ',layerpairs=' ',projpairs=' ',datatype='Correlation'):
    if np.max(binsdRF)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 65
        dim12label = 'RF (\N{DEGREE SIGN})'

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    fig,ax    = plt.subplots(1,1,figsize=(3,3))
    ilp = 0
    ipp = 0
    handles = []
    for iap,areapair in enumerate(areapairs):
        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        handles.append(shaded_error(x=binsdRF,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                        ax = ax,color=clrs_areapairs[iap],label=areapair))
        bindata = bin_dist_mean[:,iap,ilp,ipp]
        xdata = binsdRF[~np.isnan(bindata)]
        ydata = bindata[~np.isnan(bindata)]

        try:
            # slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
            # ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} linfit',linewidth=1)
            # 
            popt, pcov = curve_fit(lambda x,a,b,c: a * np.exp(-b * x) + c, xdata, ydata, p0=[ydata[-1]-ydata[0], ydata[-1]-ydata[0], ydata[-1]],bounds=(-10, 10))
            ax.plot(xdata, popt[0] * np.exp(-popt[1] * xdata) + popt[2],linestyle='--',color=clrs_areapairs[iap],label=f'{areapair} fit',linewidth=1)
        except:
            print('curve_fit failed for %s' % (areapair))
            continue
        
    ax.legend(handles=handles,labels=areapairs,frameon=False)
    ax.set_xlim([0,xylim])
    ax.set_ylim([my_floor(np.min(bin_dist_mean)*0.65,3),my_ceil(np.max(bin_dist_mean)*1.1,3)])
    ax.set_xlabel(u'Δ %s' % dim12label)   
    # ax.set_title('%s\n Joint' % (areapair),c=clrs_areapairs[iap])
    ax.set_ylabel(datatype)

    plt.tight_layout()
    return fig

def plot_corr_radial_tuning_projs(binsdRF,bin_dist_count_ses,bin_dist_data_ses,	
                           areapairs=' ',layerpairs=' ',projpairs=' ',datatype='Correlation',
                           min_counts=25):
    
    #Colors:
    clrs_areapairs      = get_clr_area_pairs(areapairs) 
    clrs_projpairs      = get_clr_labelpairs(projpairs)
    if len(projpairs)==1:
        clrs_projpairs =[clrs_projpairs]

    #Stats:
    testbins        = [[0,20],[25,70]]
    testbincolors   = ['grey','grey']
    testlabels      = ['Center','Surround']
    
    statpairs_areas = [[('unl-unl','lab-unl'),
            ('unl-unl','lab-lab'),
            ('lab-unl','lab-lab'),
            ],
            [('unl-unl','lab-unl'),
            ('unl-unl','lab-lab'),
            ('lab-unl','lab-lab'),
            ],
            [('unl-unl','lab-unl'),
            ('unl-unl','unl-lab'),
            ('unl-unl','lab-lab'),
            ('unl-lab','lab-unl'),
            ('unl-lab','lab-lab'),
            ('lab-unl','lab-lab'),
            ]] #for statistics

    # stattest = 't-test_paired'
    stattest = 'Wilcoxon'
    # multcompcorr = 'Benjamini-Hochberg'
    multcompcorr = None

    #Compute data mean and error:
    temp = copy.deepcopy(bin_dist_data_ses)
    temp[bin_dist_count_ses<min_counts] = np.nan

    data_mean   = np.nanmean(temp,axis=0)
# 
    # data_mean   = nanweightedaverage(temp, weights=bin_dist_count_ses, axis=0)
    data_error  = np.nanstd(temp,axis=0) / np.sqrt(np.sum(~np.isnan(temp),axis=0))
    # data_error  = np.nanstd(temp,axis=0) / np.sqrt(np.shape(temp)[0])

    #Make figure:
    fig,axes    = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*4,3),sharex=False,sharey=True)

    if len(areapairs)==1:
        axes = [axes]
        clrs_areapairs      = [clrs_areapairs]

    #Make stats figure:
    # fig2,axes2    = plt.subplots(2,len(areapairs),figsize=(len(areapairs)*3,6),sharex=True)
   
    # Number of bootstrap iterations
    # n_bootstrap     = 1000
    # slopedata   = np.empty((len(areapairs),len(projpairs),n_bootstrap))
    ilp = 0
    handles = []
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        areaprojpairs = projpairs.copy()
        for ipp,projpair in enumerate(projpairs):
            areaprojpairs[ipp]       = areapair.split('-')[0] + projpair.split('-')[0] + '-' + areapair.split('-')[1] + projpair.split('-')[1]

        for ipp,projpair in enumerate(projpairs):
            handles.append(shaded_error(x=binsdRF,y=data_mean[:,iap,ilp,ipp],yerror=data_error[:,iap,ilp,ipp],
                            ax = ax,color=clrs_projpairs[ipp],label=projpair))
            # bindata     = data_mean[:,iap,ilp,ipp]
            # xdata       = binsdRF[(~np.isnan(bindata)) & (binsdRF<=60)]
            # ydata       = bindata[(~np.isnan(bindata)) & (binsdRF<=60)]
            # countdata   = bin_dist_count_ses[(~np.isnan(bindata)) & (binsdRF<=60),0,0,0].astype(int)
            # countdata   = np.clip(countdata,a_min=0,a_max=1000)
            # var_y = np.tile(0.08,len(xdata))   # Bin-level variances
            # try:
            #     slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
            #     ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_projpairs[ipp],label=f'{projpair} linfit',linewidth=1)
            # except:
            #     print('curve_fit failed for %s' % (projpair))
            #     continue

        # Define the data
        data = bin_dist_data_ses[:,:, iap, ilp, :]

        # Reshape the data to a long format
        n_sessions, n_delta_rf, n_cell_types = data.shape
        data_long = np.reshape(data, (n_sessions * n_cell_types * n_delta_rf,))

        # Create a dataframe with the data
        df = pd.DataFrame({
            'correlation': data_long,
            'session': np.repeat(np.arange(n_sessions), n_delta_rf * n_cell_types),
            # 'delta_rf': np.repeat(np.arange(n_delta_rf), n_sessions * n_cell_types),
            'delta_rf': np.tile(np.repeat(np.arange(n_delta_rf),n_cell_types),n_sessions),
            'labeled': np.tile(np.arange(n_cell_types), n_sessions * n_delta_rf)
        })

        # Fit the ANOVA model
        model = ols('correlation ~ C(delta_rf) + C(labeled) + C(labeled):C(delta_rf)', data=df).fit()
        testlabels = ['Delta RF','Proj. Type','Interaction']
        
        # Perform the ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Print the ANOVA table
        print(anova_table)
        # anova_table['F'][0]
        for itest,testlabel in enumerate(testlabels):
            ax.text(0.02,1-(itest+1)*0.1,f'{testlabel}: F = {anova_table["F"][itest]:.2f}, p = {anova_table["PR(>F)"][itest]:.2f}',transform=ax.transAxes,fontsize=8,ha='left')
        # print(anova_table.to_string(formatters={'F': '%5.2f', 'PR(>F)': '%5.2f'}))

        # for i,bin in enumerate(testbins):
        #     rectmin,rectmax = np.nanpercentile(data_mean,99),np.nanpercentile(data_mean,100)
        #     ax.add_patch(Rectangle((bin[0], rectmin), bin[1]-bin[0], rectmax-rectmin, 
        #                            color=testbincolors[i], alpha=0.3, transform=ax.transData))
        #     ax.text((bin[0]+bin[1])/2, rectmin, testlabels[i], ha='center', va='bottom', transform=ax.transData)
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                # fancybox=True, shadow=True, ncol=5)
        ax.legend(handles=handles,labels=areaprojpairs,
                  fancybox=True, shadow=True,  loc='center left',bbox_to_anchor=(1.02, 0.5),fontsize=7)

        ax.set_xlim([0,65])
        ax.set_ylim(np.nanpercentile(data_mean,[0,100]))
        ax.set_ylim(my_floor(ax.get_ylim()[0],2),my_ceil(ax.get_ylim()[1],2))
        ax.set_yticks([ax.get_ylim()[0],np.mean(ax.get_ylim()),ax.get_ylim()[1]])
        ax.set_xlabel(u'Δ %s' % 'RF (\N{DEGREE SIGN})')   
        ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
        if iap==0:
            ax.set_ylabel(datatype)

        # for i,bin in enumerate(testbins):
        #     # rectmin,rectmax = np.nanpercentile(data_mean,99),np.nanpercentile(data_mean,100)
        #     # ax.add_patch(Rectangle((bin[0], rectmin), bin[1]-bin[0], rectmax-rectmin, 
        #     #                        color=testbincolors[i], alpha=0.3, transform=ax.transData))
        #     # ax.text((bin[0]+bin[1])/2, rectmin, testlabels[i], ha='center', va='bottom', transform=ax.transData)
        #     idx = np.logical_and(binsdRF>=bin[0],binsdRF<=bin[1])
        #     data = nanweightedaverage(bin_dist_data_ses[:,idx,:,:,:],
        #                                         bin_dist_count_ses[:,idx,:,:,:],axis=1)
        #     bin_center_count = np.nansum(bin_dist_count_ses[:,idx,:,:,:],axis=1)
        #     data[bin_center_count<min_counts] = np.nan
        #     df              = pd.DataFrame(data=data[:,iap,:,:].squeeze(),columns=projpairs)
        #     df              = df.dropna(axis=0).reset_index(drop=True) #drop occasional missing data
        #     ax = axes2[i,iap]

        #     sns.stripplot(data=df,ax=ax,palette=clrs_projpairs,legend=False)
        #     sns.lineplot(data=df.T,ax=ax,palette='gray',legend=False,linewidth=0.5,linestyle='-')
        #     ax.set_xticks(range(len(df.columns)))
        #     ax.set_xticklabels(labels=projpairs,rotation=60,fontsize=7)
        #     annotator = Annotator(ax, statpairs_areas[iap], data=df,order=list(df.columns))
        #     annotator.configure(test=stattest, text_format='star', loc='inside',line_height=0,text_offset=-0.5,fontsize=7,	
        #                         line_width=1,comparisons_correction=multcompcorr,verbose=0,
        #                         correction_format='replace')
        #     annotator.apply_and_annotate()
        #     # from scipy.stats import wilcoxon
        #     # print('wilcoxon signed rank test (unl-unl vs lab-lab), p = %1.3f' % wilcoxon(df['unl-unl'],df['lab-lab'],alternative='two-sided')[1])
        #     ax.set_title('%s - %s' % (areapair,testlabels[i]),c=clrs_areapairs[iap])
        #     if iap==0:
        #         ax.set_ylabel(datatype)

    sns.despine(fig,top=True,right=True,offset=3)
    # fig.tight_layout()
    # fig2.tight_layout()
    
    return fig#,fig2

# def plot_corr_radial_tuning_projs(binsdRF,bin_dist_count,bin_dist_data,	
#                            areapairs=' ',layerpairs=' ',projpairs=' ',datatype='Correlation'):
#     if np.max(binsdRF)>100:
#         xlim               = 250
#         dim12label = 'XY (um)'
#     else:
#         xlim               = 65
#         dim12label = 'RF (\N{DEGREE SIGN})'

#     clrs_areapairs      = get_clr_area_pairs(areapairs) 
#     clrs_projpairs      = get_clr_labelpairs(projpairs)
#     if len(projpairs)==1:
#         clrs_projpairs =[clrs_projpairs]

#     fig,axes    = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3),sharex=True,sharey=True)
#     if len(areapairs)==1:
#         axes = [axes]
#         clrs_areapairs      = [clrs_areapairs]

#     if datatype=='Correlation':
#         bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
#     elif datatype=='Fraction':
#         bin_dist_error = np.sqrt(bin_dist_data*(1-bin_dist_data)/bin_dist_count) * 2.576 #99% CI
    
#     # Number of bootstrap iterations
#     n_bootstrap     = 1000
#     slopedata   = np.empty((len(areapairs),len(projpairs),n_bootstrap))
#     ilp = 0
#     handles = []
#     for iap,areapair in enumerate(areapairs):
#         ax = axes[iap]
#         areaprojpairs = projpairs.copy()
#         for ipp,projpair in enumerate(projpairs):
#             areaprojpairs[ipp]       = areapair.split('-')[0] + projpair.split('-')[0] + '-' + areapair.split('-')[1] + projpair.split('-')[1]

#         for ipp,projpair in enumerate(projpairs):
#             handles.append(shaded_error(x=binsdRF,y=bin_dist_data[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
#                             ax = ax,color=clrs_projpairs[ipp],label=projpair))
#             bindata     = bin_dist_data[:,iap,ilp,ipp]
#             xdata       = binsdRF[(~np.isnan(bindata)) & (binsdRF<=60)]
#             ydata       = bindata[(~np.isnan(bindata)) & (binsdRF<=60)]
#             countdata   = bin_dist_count[(~np.isnan(bindata)) & (binsdRF<=60),0,0,0].astype(int)
#             countdata   = np.clip(countdata,a_min=0,a_max=1000)
#             var_y = np.tile(0.08,len(xdata))   # Bin-level variances
#             try:
#                 slope, intercept, r_value, p_value, std_err = linregress(xdata,ydata)
#                 ax.plot(xdata, intercept + slope*xdata,linestyle='--',color=clrs_projpairs[ipp],label=f'{projpair} linfit',linewidth=1)
#             except:
#                 print('curve_fit failed for %s' % (projpair))
#                 continue



#             for ibt in range(n_bootstrap):
#                 # Generate bootstrap samples for y
#                 y_bootstrap = [np.random.normal(mean, np.sqrt(var / n), size=n) 
#                             for mean, var, n in zip(ydata, var_y, countdata)]
#                 y_bootstrap_means = [np.mean(y) for y in y_bootstrap]
                
#                 # Fit a linear trend
#                 slope, intercept, _, _, _ = linregress(xdata, y_bootstrap_means)
#                 slopedata[iap,ipp,ibt] = slope

#             # Compute confidence intervals
#             # trend_ci = np.percentile(slopedata[iap,ipp,:], [2.5, 97.5])
#             # print(f"Bootstrap Trend CI: {trend_ci}")

#         # ax.legend(handles=handles,labels=areaprojpairs,frameon=False,bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=7)
#         # ax.legend(handles=handles,labels=areaprojpairs,frameon=False,loc='lower right',fontsize=7)
#         ax.legend(handles=handles,labels=areaprojpairs,frameon=False,loc='best',fontsize=7)
#         ax.set_xlim([0,xlim])
#         ax.set_ylim(np.percentile(bin_dist_data,[1,99]))
#         ax.set_xlabel(u'Δ %s' % dim12label)   
#         ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
#         if iap==0:
#             ax.set_ylabel(datatype)

#     fig.tight_layout(rect=(0,0,1,1))

#     # fig2,axes = plt.subplots(len(areapairs),1,figsize=(3,len(areapairs)*3),sharex=True)
#     # for iap,areapair in enumerate(areapairs):
#     #     ax = axes[iap]
#     #     for ipp,projpair in enumerate(projpairs):
#     #         ax.violinplot(slopedata[iap,ipp,:],showextrema=False,vert=False,color=clrs_projpairs[ipp])
#     #         # ax.set_title('%s' % (areapair),c=clrs_projpairs[ipp])
#     #     ax.set_ylabel('Slope')
#     #     ax.set_xlabel('Labelpair')
#     #     ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])

#     #     # ax.set_xlim([-0.02,0.02])
#     # fig2.tight_layout(rect=(0,0,1,1))
    
#     return fig



def plot_corr_radial_tuning_dori(binsdRF,bin_dist_count,bin_dist_data,deltaoris,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
    
    if np.max(binsdRF)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 65
        dim12label = 'RF (\N{DEGREE SIGN})'

    ndeltaoris = len(deltaoris)
    clrs_deltaoris      = get_clr_deltaoris(deltaoris)

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    # fig,axes    = plt.subplots(len(areapairs),ndeltaoris,figsize=(len(areapairs)*3,ndeltaoris*3))
    fig,axes    = plt.subplots(1,len(areapairs),figsize=(len(areapairs)*3,3))
    if len(areapairs)==1:
        axes = [axes]
    ilp = 0
    ipp = 0
    for iap,areapair in enumerate(areapairs):
        ax = axes[iap]
        handles = []
        for idOri,dOri in enumerate(deltaoris):
            handles.append(shaded_error(x=binsdRF,y=bin_dist_data[idOri,:,iap,ilp,ipp],yerror=bin_dist_error[idOri,:,iap,ilp,ipp],
                            ax = ax,color=clrs_deltaoris[idOri],label=areapair))
                            # ax = ax,color=clrs_areapairs[iap],label=areapair))
            bindata = bin_dist_data[idOri,:,iap,ilp,ipp]
            xdata = binsdRF[(~np.isnan(bindata)) & (binsdRF<=60)]
            ydata = bindata[(~np.isnan(bindata)) & (binsdRF<=60)]
        
        ax.legend(handles=handles,labels=[str(x) for x in deltaoris],frameon=False,ncol=3,fontsize=6)
        ax.set_xlim([0,xylim])
        ax.set_ylim([my_floor(np.min(bin_dist_data[:,:,iap,:,:],)*0.65,3),my_ceil(np.max(bin_dist_data[:,:,iap,:,:],)*1.1,3)])
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s' % (areapair),c=clrs_areapairs[iap])
        ax.set_ylabel('Correlation')

    plt.tight_layout()
    return fig


def plot_corr_radial_tuning_projs_dori(binsdRF,bin_dist_count,bin_dist_data,deltaoris,	
                           areapairs=' ',layerpairs=' ',projpairs=' ',min_counts=50):
    bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
    bin_dist_data[bin_dist_count<min_counts] = np.nan
    bin_dist_error[bin_dist_count<min_counts] = 0
    
    if np.max(binsdRF)>100:
        xylim               = 250
        dim12label = 'XY (um)'
    else:
        xylim               = 65
        dim12label = 'RF (\N{DEGREE SIGN})'

    ndeltaoris = len(deltaoris)
    clrs_deltaoris      = get_clr_deltaoris(deltaoris)

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]
    clrs_projpairs      = get_clr_labelpairs(projpairs)

    fig,axes    = plt.subplots(len(areapairs),ndeltaoris,figsize=(ndeltaoris*3,len(areapairs)*3))
    if len(areapairs)==1:
        axes = axes[np.newaxis,:]
    ilp = 0
    for iap,areapair in enumerate(areapairs):
        for idOri,dOri in enumerate(deltaoris):
            ax = axes[iap,idOri]
            handles = []
            for ipp,projpair in enumerate(projpairs):

                handles.append(shaded_error(x=binsdRF,y=bin_dist_data[idOri,:,iap,ilp,ipp],yerror=bin_dist_error[idOri,:,iap,ilp,ipp],
                                ax = ax,color=clrs_projpairs[ipp],label=projpair))
                # bindata = bin_dist_data[idOri,:,iap,ilp,ipp]
                # bindata[bin_dist_count<50] = np.nan

                # xdata = binsdRF[(~np.isnan(bindata)) & (binsdRF<=60)]
                # ydata = bindata[(~np.isnan(bindata)) & (binsdRF<=60)]
        
            ax.set_xlim([0,xylim])
            ax.set_ylim([my_floor(np.nanmin(bin_dist_data)*0.75,3),my_ceil(np.nanmax(bin_dist_data)*1.1,3)])
            # ax.set_ylim(np.nanpercentile(bin_dist_data,[2,99]))
            # ax.set_ylim(np.nanpercentile(bin_dist_data,[0,100]))
            if iap==0:
                ax.set_title(u'Δ Pref = %d\N{DEGREE SIGN}' % (dOri),c=clrs_deltaoris[idOri])
            
            if idOri == np.floor(ndeltaoris/2) and iap==len(areapairs)-1:
                ax.set_xlabel(u'Δ %s' % dim12label)   
                ax.legend(handles=handles,labels=projpairs,frameon=False,ncol=2,fontsize=10)

            if idOri == 0:
                ax.set_ylabel('%s' % (areapair),c=clrs_areapairs[iap])
                # ax.set_yticks([0,0.01,0.02,0.05])
            else: 
                ax.set_yticks([])
                # ax.set_ylabel('Correlation')

    plt.tight_layout()
    return fig


def plot_corr_center_tuning_projs_dori(binsdRF,bin_dist_count_oris,bin_dist_mean_oris,
                                       bin_dist_posf_oris,bin_dist_negf_oris,
                                       deltaoris,areapairs=' ',layerpairs=' ',projpairs=' '):
    data            = np.stack((bin_dist_mean_oris,bin_dist_posf_oris,bin_dist_negf_oris),axis=0)
    counts_center   = np.nansum(bin_dist_count_oris[:,binsdRF<=20,:,:,:],axis=1)
    data_center     = np.nanmean(data[:,:,binsdRF<=20,:,:,:],axis=2)
    data_error      = np.full(data_center.shape,0.08) / counts_center**0.5

    ndeltaoris = len(deltaoris)
    clrs_deltaoris      = get_clr_deltaoris(deltaoris)

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]
    clrs_projpairs      = get_clr_labelpairs(projpairs)

    fig,axes    = plt.subplots(len(areapairs),3,figsize=(3*3,len(areapairs)*3))
    if len(areapairs)==1:
        axes = axes[np.newaxis,:]
    ilp = 0
    ylabels = ['Mean Correlation','Fraction','Fraction']
    for iap,areapair in enumerate(areapairs):
        for idtype,dtype in enumerate(['Correlation','Frac. Pos','Frac. Neg']):
            ax = axes[iap,idtype]
            data
            handles = []
            for ipp,projpair in enumerate(projpairs):

                handles.append(shaded_error(x=deltaoris,y=data_center[idtype,:,iap,ilp,ipp],yerror=data_error[idtype,:,iap,ilp,ipp],
                                ax = ax,color=clrs_projpairs[ipp],label=projpair))
            ax.legend(handles=handles,labels=projpairs,frameon=False,ncol=2,fontsize=8)

            ax.set_xlim([-5,95])
            ax.set_xticks(deltaoris)
            ax.set_ylim([my_floor(np.nanmin(data_center),3),my_ceil(np.nanmax(data_center)*1.1,3)])
            ax.set_title('%s' % (dtype))
            ax.set_ylabel(ylabels[idtype])
            ax.set_xlabel('Δ Pref. Orientation (\N{DEGREE SIGN})')

    plt.tight_layout()
    return fig

def plot_mean_frac_corr_areas(bincenters_2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
                            binsdRF,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    delta_x,delta_y   = np.meshgrid(bincenters_2d,bincenters_2d)

    min_counts          = 200
    xy_min              = 10

    if np.max(bincenters_2d)>100:
        xylim               = 250
        gaussian_sigma      = 3
        dim1label = 'X (um)'
        dim2label = 'Y (um)'
        dim12label = 'XY (um)'
    else:
        xylim               = 70
        gaussian_sigma      = 2
        dim1label = ' Azimuth (\N{DEGREE SIGN})'
        dim2label = ' Elevation (\N{DEGREE SIGN})'
        dim12label = 'RF (\N{DEGREE SIGN})'

    clrs_areapairs      = get_clr_area_pairs(areapairs)
    cmaps = ['hot','Reds_r','Blues_r']
    idata_labels = ['Mean','Pos','Neg']
    if len(areapairs)==1:
        clrs_areapairs =[clrs_areapairs]

    fig,axes    = plt.subplots(len(areapairs),4,figsize=(12,len(areapairs)*3))
    ilp = 0
    ipp = 0
    for iap,areapair in enumerate(areapairs):
        for idata,data in enumerate([bin_2d_mean,bin_2d_posf,bin_2d_negf]):
            ax = axes[iap,idata]
    # for ilp,layerpair in enumerate(layerpairs):
        # for ipp,projpair in enumerate(projpairs):
            data                                            = copy.deepcopy(data[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bin_2d_count[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            ax.pcolor(delta_x,delta_y,data,vmin=np.nanpercentile(data,20),vmax=np.nanpercentile(data,99),cmap=cmaps[idata])
            ax.set_facecolor('grey')
            ax.set_title('%s\n%s' % (areapair, idata_labels[idata]),c=clrs_areapairs[iap])
            ax.set_xlim([-xylim,xylim])
            ax.set_ylim([-xylim,xylim])
            ax.set_xlabel(u'Δ %s' % dim1label)
            ax.set_ylabel(u'Δ %s' % dim2label)
        ax = axes[iap,3]

        ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('fraction', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        data_pos_error = np.sqrt(bin_dist_posf*(1-bin_dist_posf)/bin_dist_count) * 2.576 #99% CI
        data_neg_error = np.sqrt(bin_dist_negf*(1-bin_dist_negf)/bin_dist_count) * 2.576 #99% CI
        
        shaded_error(x=binsdRF,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                    ax = ax,color='k',label='mean')
        shaded_error(x=binsdRF,y=bin_dist_posf[:,iap,ilp,ipp],yerror=data_pos_error[:,iap,ilp,ipp],
                    ax = ax2,color='r',label='pos')
        shaded_error(x=binsdRF,y=bin_dist_negf[:,iap,ilp,ipp],yerror=data_neg_error[:,iap,ilp,ipp],
                    ax = ax2,color='b',label='neg')
        ax.legend(frameon=False)
        ax.set_xlim([xy_min,xylim])
        ax.set_ylim(np.percentile(bin_dist_mean[binsdRF>xy_min,iap,ilp,ipp],[0,100]))
        ax2.set_ylim([0,np.percentile(bin_dist_posf[binsdRF>xy_min,iap,ilp,ipp],100)])
        # ax.set_xlim([0,xylim])
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s\n Joint' % (areapair),c=clrs_areapairs[iap])
        ax.set_ylabel('correlation')

    plt.tight_layout()
    return fig

def plot_mean_frac_corr_projs(bincenters_2d,bin_2d_count,bin_2d_mean,bin_2d_posf,bin_2d_negf,
                            binsdRF,bin_dist_count,bin_dist_mean,bin_dist_posf,bin_dist_negf,	
                           areapairs=' ',layerpairs=' ',projpairs=' '):
    delta_x,delta_y   = np.meshgrid(bincenters_2d,bincenters_2d)

    min_counts          = 200

    if np.max(bincenters_2d)>100:
        xylim               = 250
        gaussian_sigma      = 3
        dim1label = 'X (um)'
        dim2label = 'Y (um)'
        dim12label = 'XY (um)'
    else:
        xylim               = 70
        gaussian_sigma      = 2
        dim1label = ' Azimuth (\N{DEGREE SIGN})'
        dim2label = ' Elevation (\N{DEGREE SIGN})'
        dim12label = 'RF (\N{DEGREE SIGN})'


    clrs_projpairs      = get_clr_labelpairs(projpairs)
    if len(projpairs)==1:
        clrs_projpairs =[clrs_projpairs]
    cmaps = ['hot','Reds_r','Blues_r']
    idata_labels = ['Mean','Pos','Neg']
   

    fig,axes    = plt.subplots(len(projpairs),4,figsize=(12,len(projpairs)*3))
    ilp = 0
    iap = 0
    # for iap,areapair in enumerate(areapairs):
    for ipp,projpair in enumerate(projpairs):
        for idata,data in enumerate([bin_2d_mean,bin_2d_posf,bin_2d_negf]):
            ax = axes[ipp,idata]
    # for ilp,layerpair in enumerate(layerpairs):
            data                                            = copy.deepcopy(data[:,:,iap,ilp,ipp])
            data[np.isnan(data)]                            = np.nanmean(data)
            data                                            = gaussian_filter(data,sigma=[gaussian_sigma,gaussian_sigma])
            data[bin_2d_count[:,:,iap,ilp,ipp]<min_counts]     = np.nan
            ax.pcolor(delta_x,delta_y,data,vmin=np.nanpercentile(data,20),vmax=np.nanpercentile(data,99),cmap=cmaps[idata])

            # ax.pcolor(delta_az,delta_el,data,vmin=np.nanpercentile(data,10),vmax=np.nanpercentile(data,95),cmap="crest")
            ax.set_facecolor('grey')
            ax.set_title('%s\n%s' % (projpair, idata_labels[idata]),c=clrs_projpairs[ipp])
            ax.set_xlim([-xylim,xylim])
            ax.set_ylim([-xylim,xylim])
            ax.set_xlabel(u'Δ %s' % dim1label)
            ax.set_ylabel(u'Δ %s' % dim2label)
        ax = axes[ipp,3]

        ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('fraction', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)

        bin_dist_error = np.full(bin_dist_count.shape,0.08) / bin_dist_count**0.5
        data_pos_error = np.sqrt(bin_dist_posf*(1-bin_dist_posf)/bin_dist_count) * 2.576 #99% CI
        data_neg_error = np.sqrt(bin_dist_negf*(1-bin_dist_negf)/bin_dist_count) * 2.576 #99% CI
        
        shaded_error(x=binsdRF,y=bin_dist_mean[:,iap,ilp,ipp],yerror=bin_dist_error[:,iap,ilp,ipp],
                    ax = ax,color='k',label='mean')
        shaded_error(x=binsdRF,y=bin_dist_posf[:,iap,ilp,ipp],yerror=data_pos_error[:,iap,ilp,ipp],
                    ax = ax2,color='r',label='pos')
        shaded_error(x=binsdRF,y=bin_dist_negf[:,iap,ilp,ipp],yerror=data_neg_error[:,iap,ilp,ipp],
                    ax = ax2,color='b',label='neg')

        # ax.plot(binsdRF,bin_dist_mean[:,iap,ilp,ipp],color='k',label='mean')
        # ax.plot(binsdRF,bin_dist_posf[:,iap,ilp,ipp],color='r',label='pos')
        # ax.plot(binsdRF,bin_dist_negf[:,iap,ilp,ipp],color='b',label='neg')
        ax.legend(frameon=False)
        ax.set_xlim([0,xylim])
        ax.set_xlabel(u'Δ %s' % dim12label)   
        ax.set_title('%s\n Joint' % (projpair),c=clrs_projpairs[ipp])
        ax.set_ylabel('correlation')
    plt.tight_layout()
    return fig



