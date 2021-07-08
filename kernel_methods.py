____
# Universite de Technologie de Troyes__<br/>
# Universidad Tecnologica Nacional Buenos Aires__<br/
# Martin Palazzo__<br/>
# code author: mpalazzo@frba.utn.edu.ar__<br/>
# Useful kernel methods functions



# kernel methods
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import norm


def estim_gammakernel_median(X, nb_samples_max=300):
	"""
	estimate the gamma of the kernel by computing the median distance between all samples

	"""
    m = X.shape[0]
    
    if m > nb_samples_max:
        isub   = np.random.choice(m, nb_samples_max, replace=False)
        dist_X = pdist(X[isub,:])
    else:
        dist_X = pdist(X)     
    sigma  = np.median(dist_X)
    gamma = 1/(2*(sigma**2))
    return gamma

def compute_gaussian_kernel(x,y):
	"""
	create a kernel matrix from RBF kernel
	"""
	'compute '
	k = rbf_kernel(x,y,gamma=estim_gammakernel_median(x))
	return k

def kernel_delta(x1, x2):
	"""
	create a kernel matrix from Delta kernel
	"""
    n_1 = x1.shape[1]
    n_2 = x2.shape[1]
    K = np.zeros((n_1, n_2))
    u_list = np.unique(x1)
    for ind in u_list:
        ind_1 = np.where(x1 == ind)[1]
        ind_2 = np.where(x2 == ind)[1]
        K[np.ix_(ind_1, ind_2)] = 1
    return K

def kernel_alignment(k1,k2,m):
	"""
	compute the kernel alignment between a pair of kernel matrices
	"""
    #cte = np.full((m, m),1/m)
    #idn = np.identity(m)
    #un = idn - cte
    m = np.shape(k1)[0]
    #ktest_cent = k1
    #ktarget_cent = k2
    #ktest_cent = np.dot(np.dot(un,k1),un)
    #ktarget_cent = np.dot(np.dot(un,k2),un)    
    num = np.sum(np.multiply(k1, k2))
    den = (norm(k1, ord = 'fro'))*(norm(k2, ord = 'fro'))
    #alignment = num/den    
    return num/den


def compute_mmd(x, y):
	"""
	compute the Maximum Mean Discrepancy metric between two distributions
	"""
    x_kernel = compute_gaussian_kernel(x, x)
    y_kernel = compute_gaussian_kernel(y, y)
    xy_kernel = compute_gaussian_kernel(x, y)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)
