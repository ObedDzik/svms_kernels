import numpy as np
from sklearn.metrics.pairwise import laplacian_kernel

def r_quadratic(X, Y, alpha=1.5, length_scale=1.0):
    pairwise_sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return (1.0 + pairwise_sq_dists / (2 * alpha * length_scale**2))**(-alpha)

def laplace_kernel(X1, X2):
    k = laplacian_kernel(X1,X2)
    return k

def cauchy_kernel(X1, X2):
    n1, _ = X1.shape
    n2, _ = X2.shape
    k = np.zeros((n1,n2))
    sigma = 0.5

    for i in range(n1):
        for j in range(n2):
            k[i,j] = 1/(1+(np.linalg.norm(X1[i] - X2[j])**2)/(sigma**2))

    return k