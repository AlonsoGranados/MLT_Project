import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
from scipy.special import gamma

def GP_UCB(D, mu, sigma,k,T,f):
    B = 0
    for i in range(T):
        values = mu + np.sqrt(B)*sigma
        x = np.argmax(values)

        y = np.random.randn(0,1) + f(x)

        # Bayessian update

def compute_mean_cov(X,X_star,f,K):
    inv = np.linalg.inv(K(X, X))
    K_s = K(X_star, X)
    K_ss = K(X_star, X_star)
    K_s_to_inv = np.matmul(K_s, inv)

    mu = np.matmul(K_s_to_inv,f)
    sigma = K_ss - np.matmul(K_s_to_inv, K_s.T)

    return mu,sigma

def rbf_kernel(X_p,X_q):
    K = -2 * np.matmul(X_p,X_q.T)
    K += X_p * X_p
    K += X_q.T * X_q.T
    K = K * (-1/2)
    K = np.exp(K)
    return K

def matern_kernel(X_p,X_q):
    nu = 2.5
    l = 1
    K = -2 * np.matmul(X_p,X_q.T)
    K += X_p * X_p
    K += X_q.T * X_q.T
    # distance
    d = np.sqrt(K)
    K = d * (np.sqrt(2 *nu)/l)
    K = np.power(K,nu)
    K *= kv(nu, d * (np.sqrt(2 *nu)/l))
    K = K * (1/(gamma(nu)*np.power(2,nu-1)))
    K[np.isnan(K)] = 0
    return K

def linear_kernel(X_p,X_q):
    nu = 2.5
    l = 1
    K = -2 * np.matmul(X_p,X_q.T)
    K += X_p * X_p
    K += X_q.T * X_q.T
    # distance
    d = np.sqrt(K)
    K = d * (np.sqrt(2 *nu)/l)
    K = np.power(K,nu)
    K *= kv(nu, d * (np.sqrt(2 *nu)/l))
    K = K * (1/(gamma(nu)*np.power(2,nu-1)))
    K[np.isnan(K)] = 0
    return K

x = np.linspace(-5,5)
x = np.reshape(x,(-1,1))

mu = np.zeros(50)
sigma = matern_kernel(x,x)

f = np.random.multivariate_normal(mu,sigma,10)
for i in range(10):
    plt.plot(x,f[i])
plt.show()

X = np.array([-2,0,2,4])
X = np.reshape(X,(-1,1))
f = np.sin(X)

mu,sigma = compute_mean_cov(X,x,f,matern_kernel)
mu = mu.reshape(-1)
f = np.random.multivariate_normal(mu,sigma,10)
for i in range(10):
    plt.plot(x,f[i])
plt.show()