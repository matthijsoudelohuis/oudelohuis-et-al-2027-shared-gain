
import numpy as np
import copy
from scipy.stats import zscore
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LOGR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm as SVM
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from scipy.stats import zscore, ttest_rel
from scipy import linalg

# from utils.dimreduc_lib import *
from utils.rf_lib import filter_nearlabeled
from utils.plot_lib import *


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    Filters out nans in any column
    """
    notnan = np.logical_and(~np.isnan(v1), ~np.isnan(v2))
    v1 = v1[notnan]
    v2 = v2[notnan]
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.rad2deg(angle_rad)

def angles_between(v):
    """ Returns the angle in degrees between each of the columns in vector array v:
    """
    angles = np.full((v.shape[1],v.shape[1]), np.nan)
    for i in range(v.shape[1]):
        for j in range(i+1,v.shape[1]):
            angles[i,j] = angle_between(v[:,i],v[:,j])
            angles[j,i] = angles[i,j]
    return angles


def EV(Y,Y_hat):
    # e = Y - Y_hat
    # ev = 1 - np.trace(e.T @ e) / np.trace(Y.T @ Y) #fraction of variance explained
    ev = 1 - np.nanvar(Y-Y_hat) / np.nanvar(Y)
    return ev

def var_along_dim(data,weights):
    """
    Compute the variance of the data projected onto the weights.
    
    Parameters
    ----------
    data : array (n_samples, n_features)
        Data to project
    weights : array (n_features)
        Weights for projecting the data into a lower dimensional space
    
    Returns
    -------
    ev : float
        Proportion of variance explained by the projection.
    """
    assert data.shape[1] == weights.shape[0], "data and weights must have the same number of features"
    assert weights.ndim == 1, "weights must be a vector"
    
    weights     = unit_vector(weights) # normalize weights
    var_proj    = np.var(np.dot(data, weights)) # compute variance of projected data
    var_tot     = np.var(data, axis=0).sum() # compute total variance of original data
    ev          = var_proj / var_tot # compute proportion of variance explained 
    return ev

def LM(Y, X, lam=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    # Y_hat = X @ B_hat
    return B_hat


def find_optimal_lambda(X,y,model_name='LOGR',kfold=5,clip=False):
    if model_name == 'LogisticRegression':
        model_name = 'LOGR'
    assert len(X.shape)==2, 'X must be a matrix of samples by features'
    assert len(y.shape)==1, 'y must be a vector'
    assert X.shape[0]==y.shape[0], 'X and y must have the same number of samples'
    # assert model_name in ['LOGR','SVM','LDA'], 'regularization not supported for model %s' % model_name
    assert model_name in ['LOGR','SVM','LDA','Ridge','Lasso','LinearRegression'], 'regularization not supported for model %s' % model_name

    # Define the k-fold cross-validation object
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance for each fold
    fold_performance = np.zeros((kfold,))

    # Find the optimal regularization strength (lambda)
    lambdas = np.logspace(-4, 4, 20)
    cv_scores = np.zeros((len(lambdas),))
    for ilambda, lambda_ in enumerate(lambdas):
        
        if model_name == 'LOGR':
            model = LOGR(penalty='l1', solver='liblinear', C=lambda_)
            score_fun = 'accuracy'
        elif model_name == 'SVM':
            model = SVM.SVC(kernel='linear', C=lambda_)
            score_fun = 'accuracy'
        elif model_name == 'LDA':
            n_components = np.unique(y).size-1
            model = LDA(n_components=n_components,solver='eigen', shrinkage=np.clip(lambda_,0,1))
            score_fun = 'accuracy'
        elif model_name in ['Ridge', 'Lasso']:
            model = getattr(sklearn.linear_model,model_name)(alpha=lambda_)
            score_fun = 'r2'
        elif model_name in ['ElasticNet']:
            model = getattr(sklearn.linear_model,model_name)(alpha=lambda_,l1_ratio=0.9)
            score_fun = 'r2'

        scores = cross_val_score(model, X, y, cv=kf, scoring=score_fun)
        cv_scores[ilambda] = np.mean(scores)
    optimal_lambda = lambdas[np.argmax(cv_scores)]
    # print('Optimal lambda for session %d: %0.4f' % (ises, optimal_lambda))
    if clip:
        optimal_lambda = np.clip(optimal_lambda, 0.03, 166)
    # optimal_lambda = 1
    return optimal_lambda

def circular_abs_error(y_true, y_pred):
    # y_true and y_pred in degrees (0-360)
    error = np.abs((y_pred - y_true + 180) % 360 - 180)
    return np.mean(error)  # or np.median(error)

def my_decoder_wrapper(Xfull,Yfull,model_name='LOGR',kfold=5,lam=None,subtract_shuffle=True,
                          scoring_type=None,norm_out=False,n_components=None):
    if model_name == 'LogisticRegression':
        model_name = 'LOGR'
    assert len(Xfull.shape)==2, 'Xfull must be a matrix of samples by features'
    assert len(Yfull.shape)==1, 'Yfull must be a vector'
    assert Xfull.shape[0]==Yfull.shape[0], 'Xfull and Yfull must have the same number of samples'
    assert model_name in ['LOGR','SVM','LDA','Ridge','Lasso','LinearRegression','SVR'], 'regularization not supported for model %s' % model_name
    assert lam is None or lam > 0
    
    if lam is None:
        lam = find_optimal_lambda(Xfull,Yfull,model_name=model_name,kfold=kfold)

    if model_name == 'LOGR':
        model = LOGR(penalty='l1', solver='liblinear', C=lam)
    elif model_name == 'SVM':
        model = SVM.SVC(kernel='linear', C=lam)
    elif model_name == 'LDA':
        if n_components is None: 
            n_components = np.unique(Yfull).size-1
        model = LDA(n_components=n_components,solver='eigen', shrinkage=np.clip(lam,0,1))
        # model = LDA(n_components=n_components,solver='svd')
    elif model_name == 'GBC': #Gradient Boosting Classifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=10, random_state=0,max_features='sqrt')
    elif model_name in ['Ridge', 'Lasso']:
        model = getattr(sklearn.linear_model,model_name)(alpha=lam)
    elif model_name in ['ElasticNet']:
        model = getattr(sklearn.linear_model,model_name)(alpha=lam,l1_ratio=0.9)
    elif model_name == 'SVR':
        from sklearn.svm import SVR
        model = SVR(kernel='rbf',C=lam)  # or 'linear'

    if scoring_type is None:
        scoring_type = 'accuracy_score' if model_name in ['LOGR','SVM','LDA','GBC'] else 'r2_score'
    
    if scoring_type == 'circular_abs_error':
        score_fun           = circular_abs_error
    else: 
        score_fun           = getattr(sklearn.metrics,scoring_type)

    # Define the number of folds for cross-validation
    kf = KFold(n_splits=kfold, shuffle=True, random_state=0)

    # Initialize an array to store the decoding performance
    performance         = np.full((kfold,), np.nan)
    performance_shuffle = np.full((kfold,), np.nan)
    # weights             = np.full((kfold,np.shape(Xfull)[1]), np.nan) #deprecated, estimate weights from all data, not cv
    projs               = np.full((np.shape(Xfull)[0]), np.nan)

    # Loop through each fold
    for ifold, (train_index, test_index) in enumerate(kf.split(Xfull)):
        # Split the data into training and testing sets
        X_train, X_test = Xfull[train_index], Xfull[test_index]
        y_train, y_test = Yfull[train_index], Yfull[test_index]

        # Train a classification model on the training data with regularization
        model.fit(X_train, y_train)

        # weights[ifold,:] = model.coef_ #deprecated, estimate weights from all data, not cv

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the decoding performance for this fold
        performance[ifold] = score_fun(y_test, y_pred)
        projs[test_index] = y_pred

        if subtract_shuffle:
            # Shuffle the labels and calculate the decoding performance for this fold
            np.random.shuffle(y_train)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            performance_shuffle[ifold] = score_fun(y_test, y_pred)

    if subtract_shuffle: # subtract the shuffling performance from the average perf
        performance_avg = np.mean(performance - performance_shuffle)
    else: # Calculate the average decoding performance across folds
        performance_avg = np.mean(performance)
    if norm_out: # normalize to maximal range of performance (between shuffle and 1)
        performance_avg = performance_avg / (1-np.mean(performance_shuffle))

    #Estimate the weights from the entire dataset:
    model.fit(Xfull,Yfull)
    if hasattr(model,'coef_'):
        weights = model.coef_.ravel()
    else:
        weights = []

    if np.shape(Xfull)[1] == np.shape(weights)[0]:
    # if len(np.unique(Yfull)) == 2:
        ev      = var_along_dim(Xfull,weights)
    else:
        ev = None
        # ev      = var_along_dim(Xfull,weights)

    return performance_avg,weights,projs,ev

def prep_Xpredictor(X,y):
    X           = zscore(X, axis=0,nan_policy='omit')
    idx_nan     = ~np.all(np.isnan(X),axis=1)
    X           = X[idx_nan,:]
    y           = y[idx_nan]
    X[:,np.all(np.isnan(X),axis=0)] = 0
    X           = np.nan_to_num(X,nan=np.nanmean(X,axis=0,keepdims=True))
    y           = np.nan_to_num(y,nan=np.nanmean(y,axis=0,keepdims=True))
    return X,y,idx_nan

def balance_trial(X,y,sample_min_trials=20):
    N0,N1 = np.sum(y==0),np.sum(y==1)
    mintrials =  np.min([N0,N1])
    if mintrials < sample_min_trials:
        idx0 = np.random.choice(np.where(y==0)[0],size=sample_min_trials,replace=True)
        idx1 = np.random.choice(np.where(y==1)[0],size=sample_min_trials,replace=True)
        yb = np.concatenate((y[idx0],y[idx1]))
        Xb = np.concatenate((X[idx0,:],X[idx1,:]))
    else: 
        idx0 = np.random.choice(np.where(y==0)[0],size=mintrials,replace=False)
        idx1 = np.random.choice(np.where(y==1)[0],size=mintrials,replace=False)
        yb  = np.concatenate((y[idx0],y[idx1]))
        Xb  = np.concatenate((X[idx0,:],X[idx1,:]))
    return Xb,yb    
