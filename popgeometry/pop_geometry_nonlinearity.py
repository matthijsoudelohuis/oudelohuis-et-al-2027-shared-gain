#%% 
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from sklearn.decomposition import PCA

from utils.gain_lib import * 
from utils.plot_lib import * #get all the fixed color schemes
from utils.nonlin_lib import * 
from utils.params import params

figdir = os.path.join(params['figdir'],'popgeometry','nonlinearTF')

#%%
######   #####     #       ####### #     #    #     # ####### ######  ####### #       
#     # #     #   # #      #     # ##    #    ##   ## #     # #     # #       #       
#     # #        #   #     #     # # #   #    # # # # #     # #     # #       #       
######  #       #     #    #     # #  #  #    #  #  # #     # #     # #####   #       
#       #       #######    #     # #   # #    #     # #     # #     # #       #       
#       #     # #     #    #     # #    ##    #     # #     # #     # #       #       
#        #####  #     #    ####### #     #    #     # ####### ######  ####### ####### 

#%%  
nNeurons        = 1000
nTrials         = 3200

noris           = 8
tuning_level    = 1
popmodulation_level = 1
noise_std       = 0.15

oris            = np.linspace(0,360,noris+1)[:-1]
locs            = np.random.rand(nNeurons) * np.pi * 2  # circular mean
kappa           = 2  #tuning concentration parameter (higher = more sharply tuned)

tuning_var      = np.random.rand(nNeurons) * tuning_level #how strongly tuned neurons are
popcouplings    = np.random.rand((nNeurons))
poprates        = np.random.rand((nTrials)) * popmodulation_level

ori_trials      = np.random.choice(oris,nTrials)

# Format: (name, nl_func, n_shape, p0_shape, bounds_shape)
# Responses are min-max normalised to [0,1] before fitting, so all nonlinearities
# operate in the same output regime without per-model gain/offset parameters.
# NL_CONFIGS = [
#     ('Linear',          nl_linear,   0, [],      []),
#     ('ReLU',            nl_relu,     0, [],      []),
#     ('Softplus',        nl_softplus, 1, [5.0],   [(0.01, 50.0)]),
#     # ('Tanh',            nl_tanh,     1, [1.0],   [(0.0, None)]),
#     ('Exp',             nl_exp,      0, [],      []),
#     ('Power-law (p=2)', nl_powerlaw, 1, [2.0],   [(0.1,  4.0)]),
#     ('Sigmoid',         nl_sigmoid,  1, [1],   [(0.0, None)]),
# ]

operating_range = np.array([[0,1],
                            [-0.5,1],
                            [-3,3],
                            [-1,1],
                            [-2.5,2.5],
                            [-5,3]])

nl_names = [c[0] for c in NL_CONFIGS]
nNL      = len(NL_CONFIGS)

R = np.empty((nNeurons,nTrials))
for iN in range(nNeurons):
    tuned_resp = vonmises.pdf(np.deg2rad(ori_trials), loc=locs[iN], kappa=kappa)
    R[iN,:] = (tuned_resp / np.max(tuned_resp)) * tuning_var[iN]

inputmat = np.full((nNeurons,nTrials),np.nan)
for iN in range(nNeurons):
    inputmat[iN,:] = R[iN,:] + poprates * popcouplings[iN]

for inonlin, (name, nl_func, n_shape, p0_shape, bounds_shape) in enumerate(NL_CONFIGS):
    #Scale the input to cover the operating range of this transfer function:
    scaledinputmat = copy.deepcopy(inputmat)
    scaledinputmat /= np.max(scaledinputmat) 
    scaledinputmat *= (operating_range[inonlin,1] -  operating_range[inonlin,0])
    scaledinputmat += operating_range[inonlin,0]

    respmat = np.full((nNeurons,nTrials),np.nan)
    for iN in range(nNeurons):
        respmat[iN,:] = simulate_responses(scaledinputmat[iN,:], nl_func, noise_std=noise_std *  (operating_range[inonlin,1] -  operating_range[inonlin,0]), n_trials=1)

    pca         = PCA(n_components=15) #construct PCA object with specified number of components
    Xp          = pca.fit_transform(respmat.T).T #fit pca to response matrix (n_samples by n_features)
    #dimensionality is now reduced from N by K to ncomp by K
    
    oris        = np.sort(np.unique(ori_trials))

    ori_ind     = [np.argwhere(np.array(ori_trials) == iori)[:, 0] for iori in oris]

    shade_alpha = 0.2
    lines_alpha = 0.8
    pal = sns.color_palette('husl', len(oris))
    pal = np.tile(sns.color_palette('husl', int(len(oris))), (2, 1))
    
    # projections = [(0, 1), (1, 2), (0, 2)]
    projections = [(0, 1), (1, 2)]
    fig, axes = plt.subplots(1, len(projections), figsize=[
                            len(projections)*3, 3], 
                            sharey=False, sharex=False)
    for ax, proj in zip(axes, projections):
        # plot orientation separately with diff colors
        for t, t_type in enumerate(oris):
            # get all data points for this ori along first PC or projection pairs
            x = Xp[proj[0], ori_ind[t]]
            y = Xp[proj[1], ori_ind[t]]  # and the second
            # each trial is one dot
            ax.scatter(x, y, color=pal[t], s=10, alpha=0.8)
            # ax.scatter(x, y, color=pal[t], s=ses.respmat_videome[ori_ind[t]], alpha=0.8)     #each trial is one dot
            ax.set_xlabel('PC {}'.format(proj[0]+1))  # give labels to axes
            ax.set_ylabel('PC {}'.format(proj[1]+1))

        sns.despine(fig=fig, top=True, right=True)

    # Put a legend to the right of the current axis
    # ax.legend(oris, title='Orientation', frameon=False, fontsize=6, title_fontsize=8,
    #         loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(name)
    plt.tight_layout()
    my_savefig(fig,figdir,f'PCA_Nonlinearity_{name}')


#%%  
ncomps = 15

fig, axes = plt.subplots(1, 1, figsize=[4*cm,4*cm])
ax = axes
for inonlin, (name, nl_func, n_shape, p0_shape, bounds_shape) in enumerate(NL_CONFIGS):
    #Scale the input to cover the operating range of this transfer function:
    scaledinputmat = copy.deepcopy(inputmat)
    scaledinputmat /= np.max(scaledinputmat) 
    scaledinputmat *= (operating_range[inonlin,1] -  operating_range[inonlin,0])
    scaledinputmat += operating_range[inonlin,0]

    respmat = np.full((nNeurons,nTrials),np.nan)
    for iN in range(nNeurons):
        respmat[iN,:] = simulate_responses(scaledinputmat[iN,:], nl_func, noise_std=noise_std *  (operating_range[inonlin,1] -  operating_range[inonlin,0]), n_trials=1)

    pca         = PCA(n_components=ncomps) #construct PCA object with specified number of components
    pca         = pca.fit(respmat.T) #fit pca to response matrix (n_samples by n_features)

    #dimensionality is now reduced from N by K to ncomp by K
    ax.plot(pca.explained_variance_ratio_,label=name,marker='o',markersize=3,alpha=0.8)
ax.set_xticks(range(ncomps))
ax.grid()
ax.legend(nl_names,frameon=False,bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
ax.set_xlabel('PCA component')
ax.set_ylabel('Variance')
sns.despine(fig=fig,top=True)
my_savefig(fig,figdir,f'PCA_EV_TransferFunctions')

#%% 


