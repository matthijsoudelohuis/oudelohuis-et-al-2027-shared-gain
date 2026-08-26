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
from utils.explorefigs import plot_PCA_gratings,plot_PCA_gratings_3D

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

drive_means =[10,-.15,-.5,-.25,-.5,-4]
drive_stds = [1,1,1,1,3,1]

for inonlin, (name, nl_func, n_shape, p0_shape, bounds_shape) in enumerate(NL_CONFIGS):
    print(name)
    model_ses =  generate_nonlin_data(nonlin=name,tuning_level=1,popmodulation_level=1,noise_std=0.5,
                                      drive_mean=drive_means[inonlin],drive_std=drive_stds[inonlin])
    # model_ses =  generate_nonlin_data(nonlin='Sigmoid',tuning_level=.5,popmodulation_level=.05,noise_std=.05)
    # model_ses =  generate_nonlin_data(nonlin='Exp',tuning_level=1,popmodulation_level=.5,noise_std=.5)
    # sessions = [generate_nonlin_data(nonlin='Exp',tuning_level=1,popmodulation_level=5,noise_std=.6)]
    # fig = plot_PCA_gratings(model_ses)
    # plt.hist(model_ses.respmat.flatten(),bins=100)
    fig = plot_PCA_gratings_3D(model_ses)
    # fig.suptitle(name)
    # my_savefig(fig=fig,figdir=figdir,filename='pop_geometry_synthetic_nonlinearity_%s' % name)

#%%  
ncomps = 15

fig, axes = plt.subplots(1, 1, figsize=[4*cm,4*cm])
ax = axes
for inonlin, (name, nl_func, n_shape, p0_shape, bounds_shape) in enumerate(NL_CONFIGS):
    model_ses =  generate_nonlin_data(nonlin=name,tuning_level=1,popmodulation_level=1,noise_std=0.5,
                                          drive_mean=drive_means[inonlin],drive_std=drive_stds[inonlin])
    
    pca         = PCA(n_components=ncomps) #construct PCA object with specified number of components
    pca         = pca.fit(model_ses.respmat.T) #fit pca to response matrix (n_samples by n_features)

    #dimensionality is now reduced from N by K to ncomp by K
    ax.plot(pca.explained_variance_ratio_,label=name,marker='o',markersize=3,alpha=0.8)
ax.set_xticks(range(ncomps))
ax.grid()
ax.legend(nl_names,frameon=False,loc='upper right')
ax.set_xlabel('PCA component')
ax.set_ylabel('Variance')
sns.despine(fig=fig,top=True)
# my_savefig(fig,figdir,f'PCA_EV_TransferFunctions')

#%% 


