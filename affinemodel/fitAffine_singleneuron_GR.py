#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress
import statsmodels.formula.api as smf

from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.plot_lib import * #get all the fixed color schemes
from utils.params import params

figdir = os.path.join(params['figdir'],'affinemodel')

#%% Load an example session: 
data = np.load(os.path.join(os.getcwd(),'datasets','dataset_A_1_exampleses' + '.npz'),allow_pickle=True)
sessions = data['sessions']

#%%
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% 
N               = len(celldata)

modelversions = ['all','area','plane','radius_50','radius_100','radius_500','radius_1000']
modelversions = ['random','runspeed','videome','all','area','plane','radius_50','radius_100','radius_500','radius_1000']
modelversions = ['random','area','allfast']
nmodels         = len(modelversions)
modelcoefs      = np.full((nmodels,N,3),np.nan)
model_R2        = np.full((nmodels,N),np.nan)

for imodelversion,modelversion in enumerate(modelversions):
    sessions = fitAffine_GR_singleneuron_full(sessions,
                                              modelversion=modelversion,recompute_poprate=True)
    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    modelcoefs[imodelversion,:,0] = celldata['aff_alpha_grfull']
    modelcoefs[imodelversion,:,1] = celldata['aff_beta_grfull']
    modelcoefs[imodelversion,:,2] = celldata['aff_offset_grfull']
    model_R2[imodelversion,:] = celldata['aff_r2_grfull']

#%%
sessions = fitAffine_GR_singleneuron_full(sessions,modelversion='allfast')

#%%
idx_N = ses.celldata['tuning_var'] > 0.01
# idx_N = ses.celldata['OSI'] > 0.5

fig,axes = plt.subplots(1,2+nmodels,figsize=(10+5*nmodels,5))
axes[0].imshow(Y[idx_N,:],aspect='auto',vmin=-0.5,vmax=0.5)
axes[1].imshow(T[idx_N,:],aspect='auto',vmin=-0.5,vmax=0.5)
axes[0].set_title('Raw')
axes[1].set_title('Tuned')

for i in range(nmodels):
    axes[i+2].imshow(Y_hat[idx_N,:,i],aspect='auto',vmin=-0.5,vmax=0.5)
    axes[i+2].set_title(modelversions[i])
# my_savefig(fig,figdir,'Heatmap_AffineModel_SingleNeuron_GR_%ssession' % ses.session_id,formats=['png'])

#%% 
idx_N = ses.celldata['tuning_var'] > 0.025

clrs = sns.color_palette('colorblind',nmodels)
fig,ax = plt.subplots(1,1,figsize=(nmodels*0.8,3))
sns.violinplot(data=model_R2[:,idx_N].T,ax=ax,palette=clrs,inner="box",scale='width',linewidth=1,cut=0)
for r2 in np.linspace(0,1,6):
    ax.axhline(r2,color='black',linestyle='--',alpha=0.5,zorder=-1)
ax.set_ylabel('R2')
ax.set_title('R2 values for the different models')
ax.set_xticks(range(nmodels))
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
ax.set_xticklabels([v if i != np.argmax(np.nanmean(model_R2[:,idx_N],axis=1)) else '*'+v for i,v in enumerate(modelversions)],
                   rotation=45,ha='right',fontsize=9)
# my_savefig(fig,figdir,'R2_AffineModel_SingleNeuron_GR_%ssession' % ses.session_id,formats=['png'])

print('Mean R2 for:')
for i in range(nmodels):
    idx_N = ses.celldata['tuning_var'] > 0.025
    idx_N = ses.celldata['tuning_var'] > 0
    print('%s: %.2f' % (modelversions[i],np.nanmean(model_R2[i,idx_N])))
    # print('%s: %.2f' % (modelversions[i],np.nanmean(model_R2[i,:])))
    # print('Median R2 for %s: %.2f' % (modelversions[i],np.nanmedian(model_R2[i,:])))

#%%
sessions = fitAffine_GR_singleneuron_split(sessions,modelversion='allfast',perc=50,recompute_poprate=True)

#%% 
ises = 0
fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
ax = axes
scatter_alphabeta(ax,sessions[ises].celldata)

#%% 
ises = 0
fig,axes = plt.subplots(1,2,figsize=(8,4))
sns.histplot(data=sessions[ises].celldata,x='aff_alpha_grsplit',color='green',element="step",
             common_norm=False,ax=axes[0],stat="density",hue='arealabel')
sns.histplot(data=sessions[ises].celldata,x='aff_beta_grsplit',color='blue',element="step",
             common_norm=False,ax=axes[1],stat="density",hue='arealabel')
axes[0].set_title('Mult')
axes[1].set_title('Add')
# my_savefig(fig,figdir,'AffineModelCoefHist_SingleNeuron_GR_%ssession' % sessions[ises].session_id,formats=['png'])


#%% 
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)


#%% Show histograms of the coefficients for each area
arealabels      = ['V1','PM']
bins            = np.linspace(-1,10,100)
fig,axes        = plt.subplots(1,2,figsize=(6,3))
# arealabelpairs = [['V1','PM']]

for imod,mod in enumerate(['alpha','beta']):
    ax = axes[imod]
    idx_N = np.all((celldata['roi_name'].isin(arealabels),
                    # celldata['OSI']>0.5,
                    ),axis=0)
    sns.histplot(data=celldata[idx_N],x='aff_%s_grsplit' % mod,element="step",common_norm=False,ax=ax,fill=False,
                bins=bins,stat="probability",hue='roi_name',hue_order=arealabels,palette=get_clr_areas(arealabels),cumulative=True)
    for ialp,arealabel in enumerate(arealabels):
        idx_N_area = celldata['roi_name']==arealabel
        ax.plot(np.nanmean(celldata['aff_%s_grsplit' % mod][idx_N_area]),1,markersize=7,
                    color=get_clr_areas([arealabel]),marker='v')
    if imod==0:
        ax.axvline(1,ls='--',color='k',alpha=0.5)
    # Fit linear mixed effects model on celldata ['Affine_Mult'] with 'session_id' as 
    model = smf.mixedlm("aff_%s_grsplit ~ roi_name" % mod, data=celldata[idx_N], groups=celldata["session_id"][idx_N])
    result = model.fit(reml=False)
    # print(result.summary())
    
    pval = result.pvalues[1]
    print('P-value %s (%s): %1.5f' % ('_'.join(arealabels),mod,pval))
    
    # Calculate effect size (Cohen's d)
    effect_size = result.params[1] / np.sqrt(result.cov_params().iloc[1, 1])
    print('Effect size %s (%s): %1.5f' % ('_'.join(arealabels), mod, effect_size))

    ax.text(0.75,0.5,'p=%1.4f' % (pval),ha='center',va='center',transform=ax.transAxes)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
# my_savefig(fig,figdir,'AffineModel_Areas_CoefHist_SingleNeuron_GR_%d' % nSessions,formats=['png'])

#%% Show histograms of the coefficients for each area
arealabels      = ['V1unl','V1lab','PMunl','PMlab']
bins            = np.linspace(-1,10,100)

fig,axes        = plt.subplots(2,2,figsize=(6,6))
arealabelpairs = [['V1unl','V1lab'],['PMunl','PMlab']]

for ialp,alp in enumerate(arealabelpairs):
    for imod,mod in enumerate(['alpha','beta']):
        ax = axes[ialp,imod]
        idx_N = np.all((celldata['arealabel'].isin(alp),
                        # celldata['OSI']>0.5,
                        # celldata['aff_r2_grsplit']>0.2,
                        celldata['noise_level']<20),axis=0)
        sns.histplot(data=celldata[idx_N],x='aff_%s_grsplit' % mod,element="step",common_norm=False,ax=ax,fill=False,
                    bins=bins,stat="probability",hue='arealabel',hue_order=alp,palette=get_clr_area_labeled(alp),cumulative=True)

        for ial,arealabel in enumerate(alp):
            idx_N_area = celldata['arealabel']==arealabel
            ax.plot(np.nanmean(celldata['aff_%s_grsplit' % mod][idx_N_area]),1,markersize=7,
                        color=get_clr_area_labeled([arealabel]),marker='v')
        
        # Fit linear mixed effects model on celldata ['Affine_Mult'] with 'session_id' as 
        model = smf.mixedlm("aff_%s_grsplit ~ arealabel" % mod, data=celldata[idx_N], groups=celldata["session_id"][idx_N])
        result = model.fit(reml=False)
        
        pval = result.pvalues[1]
        print('P-value %s (%s): %1.5f' % ('_'.join(arealabels),mod,pval))
        
        # Calculate effect size (Cohen's d)
        effect_size = result.params[1] / np.sqrt(result.cov_params().iloc[1, 1])
        print('Effect size %s (%s): %1.5f' % ('_'.join(arealabels), mod, effect_size))

        ax.text(0.75,0.5,'p=%1.4f' % (pval),ha='center',va='center',transform=ax.transAxes)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,figdir,'AffineModel_Labeled_CoefHist_SingleNeuron_GR_%d' % nSessions,formats=['png'])


#%% 
fig,ax = plt.subplots(1,1,figsize=(4,4))
sns.regplot(data=celldata,x='Affine_Mult',y='Affine_Add',ax=ax,color='green',
            scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})

#%% 
nbins = 25
yvars = ['Affine_Mult','Affine_Add']
# xvars = ['depth','noise_level','OSI','tuning_var','gOSI','pop_coupling','pref_ori','event_rate']
xvars = ['depth','noise_level','OSI','pop_coupling','pref_ori','event_rate']
celldata['noise_level'] = np.clip(celldata['noise_level'],0,100)
fig,axes = plt.subplots(len(yvars),len(xvars),figsize=(len(xvars)*3,len(yvars)*3))
for iy,yvar in enumerate(yvars):
    for ix,xvar in enumerate(xvars):
        # sns.regplot(data=celldata,x=xvar,y=yvar,ax=axes[iy,ix],color='black', ci=None,
                    # scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})
        sns.regplot(data=celldata,x=xvar,y=yvar,ax=axes[iy,ix],color='blue', ci=95,
                    # x_bins=np.linspace(np.nanpercentile(celldata[xvar],1),np.nanpercentile(celldata[xvar],99), nbins),
                    # x_bins=np.linspace(np.nanmin(celldata[xvar]),np.nanmax(celldata[xvar]), nbins),
                    x_bins=np.nanpercentile(celldata[xvar], np.linspace(0, 100, nbins)),
                    scatter_kws={'s': 5, 'alpha':0.5,'edgecolor': 'blue'})
                    # scatter_kws={'s': 1, 'alpha':0.1,'facecolor': 'w'})
my_savefig(fig,figdir,'AffineModel_VarCorrs_GR_%d' % nSessions,formats=['png'])

#%%
idx_N = np.all((
                celldata['aff_r2_grsplit']>0.1,
                celldata['noise_level']<20),axis=0)

# sns.scatterplot(data=celldata[idx_N],x='OSI',y='aff_alpha_grsplit')
sns.scatterplot(data=celldata[idx_N],x='OSI',y='aff_beta_grsplit')
sns.scatterplot(data=celldata[idx_N],x='gOSI',y='aff_beta_grsplit')
sns.scatterplot(data=celldata[idx_N],x='tuning_var',y='aff_beta_grsplit')

sns.scatterplot(data=celldata[idx_N],x='tuning_var',y='aff_alpha_grsplit')

#%% 
area = 'V1'
# area = 'PM'
idx_N = np.all((
                celldata['aff_r2_grsplit']>0,
                celldata['noise_level']<20,
                celldata['roi_name'].isin([area]),
                ),axis=0)

model           = 'depth + OSI + C(labeled, Treatment("unl"))'
# model           = 'roi_name + depth + OSI + C(labeled, Treatment("unl")) + pref_ori + meanF + event_rate'

mod             = 'alpha'
model_alpha     = smf.mixedlm("aff_%s_grsplit ~ %s" % (mod,model), data=celldata[idx_N], groups=celldata["session_id"][idx_N])
result_alpha    = model_alpha.fit(reml=False)
print(result_alpha.summary())

mod             = 'beta'
model_beta      = smf.mixedlm("aff_%s_grsplit ~ %s" % (mod,model), data=celldata[idx_N], groups=celldata["session_id"][idx_N])
result_beta     = model_beta.fit(reml=False)
print(result_beta.summary())


fig, axes = plt.subplots(2,2,figsize=(15,5))
for itab in range(2):
    ax = axes[itab,0]
    ax.axis('off')
    ax.axis('tight')
    if itab == 0: 
        ax.set_title('Alpha')
    ax.table(cellText=result_alpha.summary().tables[itab].values.tolist(),
             rowLabels=result_alpha.summary().tables[itab].index.tolist(),
             colLabels=result_alpha.summary().tables[itab].columns.tolist(),
             loc="center",fontsize=8)

for itab in range(2):
    ax = axes[itab,1]
    ax.axis('off')
    ax.axis('tight')
    if itab == 0: 
        ax.set_title('Beta')
    ax.table(cellText=result_beta.summary().tables[itab].values.tolist(),
             rowLabels=result_beta.summary().tables[itab].index.tolist(),
             colLabels=result_beta.summary().tables[itab].columns.tolist(),
             loc="center",fontsize=8)
fig.tight_layout()
# my_savefig(fig,figdir,'AffineModel_table_%s_GR_%dsessions' % (area,nSessions),formats=['png'])


#%% 
fig, axes = plt.subplots(1,2,figsize=(6,3))
depth_bins = np.linspace(0, 500, 500//50)
aff_alpha_mean = np.full_like(depth_bins,np.nan)
aff_alpha_err = np.full_like(depth_bins,np.nan)
aff_beta_mean = np.full_like(depth_bins,np.nan)
aff_beta_err = np.full_like(depth_bins,np.nan)

for i,depth_bin in enumerate(depth_bins):
    idx_depth = np.abs(celldata['depth']-depth_bin) < (depth_bins[1]-depth_bins[0])
    if np.sum(idx_depth) > 500: 
        aff_alpha_mean[i] = np.nanmean(celldata['aff_alpha_grsplit'][idx_depth])
        aff_alpha_err[i] = np.nanstd(celldata['aff_alpha_grsplit'][idx_depth]) / np.sqrt(np.sum(~np.isnan(celldata['aff_alpha_grsplit'][idx_depth])))
        aff_beta_mean[i] = np.nanmean(celldata['aff_beta_grsplit'][idx_depth])
        aff_beta_err[i] = np.nanstd(celldata['aff_beta_grsplit'][idx_depth]) / np.sqrt(np.sum(~np.isnan(celldata['aff_beta_grsplit'][idx_depth])))

ax=axes[0]
ax.errorbar(aff_alpha_mean,depth_bins,xerr=aff_alpha_err,fmt='o-',markerfacecolor='k',linewidth=2,color='k')
ax.set_ylabel('Depth (micrometers)')
ax.set_xlabel('Affine Alpha')
ax.set_xlim([0,2])
ax.set_ylim([0,500])
ax.invert_yaxis()
ax_nticks(ax,5)
ax.axvline(1,ls='--',color='k',alpha=0.5)

ax=axes[1]
ax.errorbar(aff_beta_mean,depth_bins,xerr=aff_beta_err,fmt='o-',markerfacecolor='k',linewidth=2,color='k')
ax.set_ylabel('Depth (micrometers)')
ax.set_xlabel('Affine Beta')
ax.set_xlim([0,1.5])
ax.set_ylim([0,500])
ax.invert_yaxis()
ax_nticks(ax,5)
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,figdir,'AffineModel_depth_GR_%dsessions' % (nSessions),formats=['png'])

#%% Show histograms of the coefficients for each area

bins            = np.linspace(-0.1,0.6,1000)
fig,axes        = plt.subplots(1,2,figsize=(6,3))
arealabelpairs = [['V1unl','V1lab'],['PMunl','PMlab']]

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    idx_N = np.all((celldata['arealabel'].isin(alp),
                    # celldata['noise_level']<20,
                    ),axis=0)
    sns.histplot(data=celldata[idx_N],x='pop_coupling',element="step",common_norm=False,ax=ax,fill=False,
                bins=bins,stat="probability",hue='arealabel',hue_order=alp,palette=get_clr_area_labeled(alp),cumulative=True)

    for ial,arealabel in enumerate(alp):
        idx_N_area = celldata['arealabel']==arealabel
        ax.plot(np.nanmean(celldata['pop_coupling'][idx_N_area]),1,markersize=7,
                    color=get_clr_area_labeled([arealabel]),marker='v')
    
    # model = smf.mixedlm("pop_coupling ~ C(labeled, Treatment('unl')) + noise_level + depth", data=celldata[idx_N], groups=celldata["session_id"][idx_N])
    model = smf.mixedlm("pop_coupling ~ labeled + noise_level + depth", data=celldata[idx_N], groups=celldata["session_id"][idx_N])
    result = model.fit(reml=False)
    # print(result.summary())
    pval = result.pvalues[1]
    print('P-value %s (%s): %1.5f' % ('_'.join(alp),mod,pval))

    # Calculate effect size (Cohen's d)
    effect_size = result.params[1] / np.sqrt(result.cov_params().iloc[1, 1])
    print('Effect size %s (%s): %1.5f' % ('_'.join(alp), mod, effect_size))

    ax.text(0.75,0.5,'p=%1.4f' % (pval),ha='center',va='center',transform=ax.transAxes)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=True)
my_savefig(fig,figdir,'Pop_Coupling_Labeled_CumHist_SingleNeuron_GR_%dsessions' % nSessions,formats=['png'])


#%% 
area = 'V1'
# area = 'PM'
idx_N = np.all((
                # celldata['noise_level']<20,
                celldata['roi_name'].isin([area]),
                ),axis=0)

# model     = smf.mixedlm("pop_coupling ~ depth + C(labeled, Treatment('unl'))", data=celldata[idx_N], groups=celldata["session_id"][idx_N])
model     = smf.mixedlm("pop_coupling ~ depth + noise_level + C(labeled, Treatment('unl'))", data=celldata[idx_N], groups=celldata["session_id"][idx_N])
result    = model.fit(reml=False)
print(result.summary())

fig, axes = plt.subplots(2,1,figsize=(6,4))
for itab in range(2):
    ax = axes[itab]
    ax.axis('off')
    ax.axis('tight')
    if itab == 0: 
        ax.set_title('Alpha')
    ax.table(cellText=result.summary().tables[itab].values.tolist(),
             rowLabels=result.summary().tables[itab].index.tolist(),
             colLabels=result.summary().tables[itab].columns.tolist(),
             loc="center")
fig.tight_layout()
