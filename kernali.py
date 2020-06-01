#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 18:06:06 2018

@author: martin
"""

############################### IMPORT PACKAGES ####################################
############################### IMPORT PACKAGES ####################################
############################### IMPORT PACKAGES ####################################
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
from scipy.optimize import minimize
from scipy.linalg import norm
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
#for plotting distance matrix
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# svm imports
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import pandas as pd
import scipy as sc

#from __future__ import division, print_function, absolute_import  



############################### IMPORT DATA ####################################
############################### IMPORT DATA ####################################
############################### IMPORT DATA ####################################

random_seed = 123
sigmafeatsel= 0.1
folds = 4
regularizator = 0.0001
greedy_kernel = 'gaussian'

#dataset = 'metabol'
dataset = 'wiscon'

if dataset == 'wiscon':
    
    data = pd.DataFrame(genfromtxt('/Users/palazzom/Dropbox/doctorado/chapter04_kernel_alignment/kernel_engineering/breast_wiscon_dataii.csv', delimiter=';'))
    data = data.drop([0], axis = 0)
    columns = np.shape(data)[1]
    samples = np.shape(data)[0]

    x0 = pd.DataFrame(data.iloc[:,1:(columns)])
    y0 = data.iloc[:,0]
         
    # separate samples and labels from each class   
    xm = x0.loc[y0[:,]==-1,:]
    ym = np.full((np.shape(xm)[0],1), 1.0)
    xb = x0.loc[y0[:]==1,:]
    yb = np.full((np.shape(xb)[0],1), -1.0)
    
    # preprocess data (mean = 0 and std dev = 1)
    x = pd.DataFrame(preprocessing.scale(np.concatenate((xm,xb), axis = 0)))
    y = np.concatenate((ym,yb), axis = 0)
    
    # obtain the number of samples 'm' and features 'n'
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    mtot = np.shape(x)[0]
    
    dataran = data.sample(frac=1, random_state = random_seed)
    #xran = pd.DataFrame(preprocessing.scale( dataran.iloc[:,1:np.shape(dataran)[1]], axis = 0))
    xran = pd.DataFrame( dataran.iloc[:,1:np.shape(dataran)[1]])
    xran.columns = dataran.columns.values[1:]
    xran.index = dataran.index.values
    yran = dataran.iloc[:,0]
    
if dataset == 'metabol':
    
    computer = 1
   
    if computer == 1:
        data0 = pd.read_csv('/Users/palazzom/Dropbox/doctorado/chapter02_metabolomic_project/metabolomic_current/metabol_dataset.csv', delimiter=';', index_col=0, header=0,decimal=",")
    if computer == 0:
        data0 = pd.read_csv('/home/daniel/Dropbox/doctorado/chapter02_metabolomic_project/metabolomic_current/metabol_dataset.csv', delimiter=';', index_col=0, header=0,decimal=",")      
    if computer == 'ambient':
        data0 = pd.read_csv('/home/martin/Dropbox/doctorado/chapter02_metabolomic_project/metabolomic_current/metabol_dataset.csv', delimiter=';', index_col=0, header=0,decimal=",")
        
    data0.columns.values

    data = pd.concat([data0[data0['class'] == 'CS'],data0[data0['class'] == 'EI']])

    columns = np.shape(data)[1]
    samples = np.shape(data)[0]
    
    # assign numerical variables to hospital feature
    data['hospital'][data['hospital']=='R']=1
    data['hospital'][data['hospital']=='H']=0
    data['hospital'][data['hospital']=='CS'] = 2
    
    # assign numerical variables to gender feature
    data['Gender'][data['Gender']=='M']=1
    data['Gender'][data['Gender']=='F']=0


    x0 = data.iloc[:,1:(columns)]
    x0 = x0.drop('hospital', 1)
    x0 = x0.drop('Gender', 1)
    x0 = x0.drop('Age', 1)
    y0 = data.iloc[:,0]
         
    # separate samples and labels from each class   
    xm = x0.loc[y0[:]=='CS',:]
    ym = np.full((np.shape(xm)[0],1), 1.0)
    xb = x0.loc[y0[:]=='EI',:]
    yb = np.full((np.shape(xb)[0],1), -1.0)
    
    # preprocess data (mean = 0 and std dev = 1)
    x = pd.DataFrame(preprocessing.scale(np.concatenate((xm,xb), axis = 0)))
    x.columns = xm.columns.values
    x.index = np.concatenate((xm.index.values,xb.index.values),axis=0)
    y = np.concatenate((ym,yb), axis = 0)
    
    # obtain the number of samples 'm' and features 'n'
    mtot = np.shape(x)[0]
    m = np.shape(x)[0]
    n = np.shape(x)[1]
     
    dataran = data.sample(frac=1, random_state = random_seed)
    dataran = dataran.drop('hospital', 1)
    dataran = dataran.drop('Gender', 1)
    dataran = dataran.drop('Age', 1)

    #xran = pd.DataFrame(preprocessing.scale( dataran.iloc[:,1:np.shape(dataran)[1]], axis = 0))
    xran = pd.DataFrame(dataran.iloc[:,1:np.shape(dataran)[1]])
    xran.columns = dataran.columns.values[1:]
    xran.index = dataran.index.values
    yran = dataran['class']

    yran.loc[yran[:]=='CS'] = 1.0
    yran.loc[yran[:]=='EI'] = -1.0


############################### DEFINE KERNELS ####################################
############################### DEFINE KERNELS ####################################
############################### DEFINE KERNELS ####################################

## Linear Kernel function
def linear_kernel(X, Y):
    k = np.zeros((np.shape(X)[0],np.shape(Y)[0]))
    k =  np.dot(X.T, Y)
    return k

## Gaussian Kernel Kernel function
def gaussian_kernel(x1,x2,sigma=0.1):    
    
    gram_gaussian = np.zeros((np.shape(x1)[0], np.shape(x2)[0]))
    
    for i in range(0,np.shape(x1)[0]):
        for j in range(0,np.shape(x2)[0]):
            gram_gaussian[i, j] = np.exp(-(norm(x1[i,:]-x2[j,:])**2)/(2*sigma**2))      
    return gram_gaussian

def gaussian_kernel2(x1,x2,sig):
    from scipy.spatial.distance import cdist
    k = np.exp((-cdist(x1,x2, 'euclidean')**2/ 2*sig**2))
    return k


################### DEFINE KERNEL ALIGNMENT, MU1 AND MU2 ##########################
################### DEFINE KERNEL ALIGNMENT, MU1 AND MU2 ##########################
################### DEFINE KERNEL ALIGNMENT, MU1 AND MU2 ##########################

# Target Kernel Alignment Function
def k_alignment_class(k,y,m):
    return np.dot(np.dot(y.T, k),y)/(m*norm(k, ord = 'fro'))

# Target Kernel Alignment Function
def k_alignment_class2(k,y,m,offset):
    koff = np.full((np.shape(k)[0],np.shape(k)[1]),offset)
    return np.dot(np.dot(y.T, k+koff),y)/(m*norm(k+koff, ord = 'fro'))

######################### PREPARE TRAIN AND TEST SETS ########################

flds = 0
xtest0 = xran.iloc[np.int(flds*np.round(mtot/folds)):np.int(flds*np.round(mtot/folds)+np.round(mtot/folds)),:]
ytest = yran.iloc[np.int(flds*np.round(mtot/folds)):np.int(flds*np.round(mtot/folds)+np.round(mtot/folds)),]
xtrain0 = xran.drop(xran.index[np.int(flds*np.round(mtot/folds)):np.int(flds*np.round(mtot/folds)+np.round(mtot/folds))])
ytrain = yran.drop(yran.index[np.int(flds*np.round(mtot/folds)):np.int(flds*np.round(mtot/folds)+np.round(mtot/folds))])

#### autoscale train and test sets
xtrain = (xtrain0 - np.mean(xtrain0))/np.std(xtrain0)    
xtest = (xtest0 - np.mean(xtrain0))/np.std(xtrain0)

np.shape(xtrain)
########### COMPUTE KERNEL MATRICES AND ALIGNMENT FOR EACH FEATURE ##################

# compute 'n' alignments, alignments contains one Kernel target alignment per feature
alignments = np.zeros(n)
alignments2 = np.zeros(n)
alignments3 = np.zeros(n)

sigma_win = np.zeros(n)
a = [ pow(0.5,i/4) for i in range(0,300) ]
b = [ pow(1.5,i/4) for i in range(0,300) ]
sigma_grid = np.concatenate((np.array(a),np.array(b)))
sigma_ali = np.zeros(np.shape(sigma_grid)[0])

mtrain = np.shape(xtrain)[0]
delta_win = np.zeros(n)
delta_grid = np.arange(-3, 3, 0.01)
delta_ali = np.zeros(np.shape(delta_grid)[0])

for i in range(0,n):
    #if greedy_kernel == 'gaussian':
        for e in range(0,np.shape(sigma_grid)[0]):
            sigma_ali[e]= k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_grid[e]),ytrain,mtrain,0)
            #sigmas[e] = c[e]
        alignments[i] = np.max(sigma_ali)
        sigma_win[i] = sigma_grid[np.argmax(sigma_ali)]
        for l in range(0,np.shape(delta_grid)[0]):
            delta_ali[l]= k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_win[i]),ytrain,mtrain, delta_grid[l])
        #alignments2[i] = np.max(rhoali)
        delta_win[i] = delta_grid[np.argmax(delta_ali)]
        sigma_ali = np.zeros(np.shape(sigma_grid)[0])
        delta_ali = np.zeros(np.shape(delta_grid)[0])

        alignments2[i] = k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_win[i]),ytrain,mtrain, delta_win[i])
        
        alignments3[i] = k_alignment_class(linear_kernel(np.reshape(xtrain.iloc[:,i],(mtrain,1)).T,np.reshape(xtrain.iloc[:,i],(mtrain,1)).T),ytrain,mtrain)
#print(str(i)) 
print("Alignments array complete")

test_results = np.column_stack((alignments3, alignments2))
plt.boxplot(test_results)
plt.title('Feature Kernel Alignments per Classifier')
plt.xticks([1, 2], ['Feat. Alig. Lin. Kernel', 'Feat. Alig. Gauss. Kernel '])
plt.ylabel('Kernel Alignments')
plt.show()

################################ TEST WITH THE OPTIMIZED OFFSET

for i in range(0,n):
    #if greedy_kernel == 'gaussian':
        for e in range(0,np.shape(sigma_grid)[0]):
            sigma_ali[e]= k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_grid[e]),ytrain,mtrain,0)
            #sigmas[e] = c[e]
        alignments[i] = np.max(sigma_ali)
        sigma_win[i] = sigma_grid[np.argmax(sigma_ali)]
        for l in range(0,np.shape(delta_grid)[0]):
            delta_ali[l]= k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_win[i]),ytrain,mtrain, delta_grid[l])
        #alignments2[i] = np.max(rhoali)
        delta_win[i] = delta_grid[np.argmax(delta_ali)]
        sigma_ali = np.zeros(np.shape(sigma_grid)[0])
        delta_ali = np.zeros(np.shape(delta_grid)[0])

        alignments2[i] = k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_win[i]),ytrain,mtrain, delta_win[i])
        
        alignments3[i] = k_alignment_class(linear_kernel(np.reshape(xtrain.iloc[:,i],(mtrain,1)).T,np.reshape(xtrain.iloc[:,i],(mtrain,1)).T),ytrain,mtrain)
#print(str(i)) 
print("Alignments array complete")


################################


kfeat = 23
k13 = gaussian_kernel2(np.reshape(np.array(x.iloc[:,kfeat]),(m,1)),np.reshape(np.array(x.iloc[:,kfeat]),(m,1)), sigma_win[kfeat])


for g in range(np.shape(delta_win)[0]):
    print(delta_win[g])

alitest = np.zeros(np.shape(delta_grid)[0])
alitest0 = np.zeros(np.shape(sigma_grid)[0])


for u in range(np.shape(delta_grid)[0]):
    alitest[u] = k_alignment_class2(k13,y,m, delta_grid[u])


plt.scatter(delta_grid[:],alitest)
plt.xticks(delta_grid[:])
plt.title('KTA optimization through offset tuning')
plt.ylabel('Feature-Wise KTA')
plt.xlabel('Offset')
plt.xticks(np.arange(-3, 4, 0.5))
plt.show()

np.max(alitest)

delta_win[kfeat]
delta_grid[np.argmax(alitest)]

delta_grid[np.argmax(alitest)]

ali_sinoffset = k_alignment_class2(k13,y,m, 0)
ali_empir = k_alignment_class2(k13,y,m, delta_grid[np.argmax(alitest)])

sigma_win[kfeat]
x.iloc[:,kfeat]



def alig_opt(xfeat,y,m,sigma_win):   
    kfeat = gaussian_kernel2(np.reshape(np.array(xfeat),(m,1)),np.reshape(np.array(xfeat),(m,1)), sigma_win)
    num1 = np.sum(np.square(kfeat))
    num2 = np.dot(y.T,y)
    num3 = np.dot(np.dot(y.T, kfeat),y)
    num4 = np.sum(kfeat)
    num = -(num1*num2) + (num3*num4)    
    den1 = 2 * np.sum(kfeat)
    den2 = np.dot(y.T,y)
    den3 = m**2
    den4 = np.dot(np.dot(y.T, kfeat),y)
    den5 = np.dot(y.T,y)
    den6 = np.sum(kfeat)
    den = (den1*den2) - (den3*den4) - (den5*den6)    
    total = num / den
    ali_deriv = k_alignment_class2(kfeat,y,m, total)
    return ali_deriv, total

alig_opt(x.iloc[:,kfeat],y,m,sigma_win[kfeat])

#ali_empir / ali_deriv

################################ TEST WITH THE OPTIMIZED OFFSET

for i in range(0,n):
    #if greedy_kernel == 'gaussian':
        for e in range(0,np.shape(sigma_grid)[0]):
            sigma_ali[e]= k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_grid[e]),ytrain,mtrain,0)
            #sigmas[e] = c[e]
        alignments[i] = np.max(sigma_ali)
        sigma_win[i] = sigma_grid[np.argmax(sigma_ali)]
        for l in range(0,np.shape(delta_grid)[0]):
            delta_ali[l]= k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_win[i]),ytrain,mtrain, delta_grid[l])
        #alignments2[i] = np.max(rhoali)
        delta_win[i] = delta_grid[np.argmax(delta_ali)]
        sigma_ali = np.zeros(np.shape(sigma_grid)[0])
        delta_ali = np.zeros(np.shape(delta_grid)[0])

        alignments2[i] = k_alignment_class2(gaussian_kernel2(np.reshape(xtrain.iloc[:,i],(mtrain,1)),np.reshape(xtrain.iloc[:,i],(mtrain,1)), sigma_win[i]),ytrain,mtrain, delta_win[i])
        
        alignments3[i] = k_alignment_class(linear_kernel(np.reshape(xtrain.iloc[:,i],(mtrain,1)).T,np.reshape(xtrain.iloc[:,i],(mtrain,1)).T),ytrain,mtrain)
#print(str(i)) 
print("Alignments array complete")


################################


test_results = np.column_stack((alignments3, alignments2))
plt.boxplot(test_results)
plt.title('Feature Kernel Alignments per Classifier')
plt.xticks([1, 2], ['Feat. Alig. Lin. Kernel', 'Feat. Alig. Gauss. Kernel '])
plt.ylabel('Kernel Alignments')
plt.show()


################################## DERIVATIVE ANALYSIS 
################################## DERIVATIVE ANALYSIS 
################################## DERIVATIVE ANALYSIS 

# derivative function of kernel target alignment
def deriv_alig (k,y,m,offset):    
    koff = k + np.full((np.shape(k)[0],np.shape(k)[1]),offset)
    num1 = np.dot(y.T,y)
    den1 = m * norm(koff, ord = 'fro')
    num2 = np.dot(np.dot(y.T,koff),y) * ((offset*(m**2)) + np.sum(k) )
    den2 = m * (norm(koff, ord = 'fro'))**3     
    return ((num1/den1)-(num2/den2))

deriv_alig(k13, y, m , np.sum(total))
deriv_alig(k13, y, m , delta_grid[np.argmax(alitest)])
deriv_alig(k13, y, m , 0)

deriv_values = np.zeros(np.shape(delta_grid)[0])

for v in range(0,np.shape(delta_grid)[0]):
    deriv_values[v] = deriv_alig(k13,y,m,delta_grid[v])
    

plt.scatter(delta_grid[:],deriv_values)
plt.xticks(delta_grid[:])
plt.show()
