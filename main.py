import numpy as np
import matplotlib.pyplot as plt

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

def square_exponential_kernel(X_p,X_q):
    K = -2 * np.matmul(X_p,X_q.T)
    K += X_p * X_p
    K += X_q.T * X_q.T
    K = K * (-1/2)
    K = np.exp(K)
    return K

x = np.linspace(-5,5)
x = np.reshape(x,(-1,1))

mu = np.zeros(50)
sigma = square_exponential_kernel(x,x)

f = np.random.multivariate_normal(mu,sigma,10)
for i in range(10):
    plt.plot(x,f[i])
plt.show()

X = np.array([-2,0,2,4])
X = np.reshape(X,(-1,1))
f = np.sin(X)

mu,sigma = compute_mean_cov(X,x,f,square_exponential_kernel)
mu = mu.reshape(-1)
f = np.random.multivariate_normal(mu,sigma,10)
for i in range(10):
    plt.plot(x,f[i])
plt.show()