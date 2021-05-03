import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern

def matern_kernel(X_p,X_q):
    X_p = np.reshape(X_p, (-1, 1))
    X_q = np.reshape(X_q, (-1, 1))
    nu = 2.5
    l = 1
    k = Matern(length_scale=l ,nu = nu)
    K = k.__call__(X_p,X_q)
    return K
# a = np.array([2])
# b = np.array([2])
# print(matern_kernel(a,b))

# Compute RBF kernel between two arrays
def rbf_kernel(X_p,X_q):
    X_p = np.reshape(X_p,(-1,1))
    X_q = np.reshape(X_q, (-1, 1))
    K = -2 * np.matmul(X_p,X_q.T)
    K += X_p * X_p
    K += X_q.T * X_q.T
    K = K * (-1/2)
    K = np.exp(K)
    return K

# Compute RBF kernel between array and single instance
def tbf_vector(A_t,x):
    v = np.zeros((A_t.shape[0],1))
    for i in range(v.shape[0]):
        v[i] = rbf_single(A_t[i],x)
    return v

# Compute RBF kernel for two instances x,y
def rbf_single(x,y):
    d = np.square(x - y)
    d = d * -1/2
    d = np.exp(d)
    return d


def InfoGain(kernel_A, sigma_noise):
    # Sigma noise is the variance
    '''
    I(y_A; f_A) = 1/2 * log * det( I + sigma**(-2)*K_A )
    K_A = k(x, x'), x, x' in A
    '''

    return 0.5 * np.log(np.linalg.det(np.add(np.eye(len(kernel_A)), (sigma_noise ** -1) * kernel_A)))

# Compute near-optimal information gain
def greedy_information_gain(A,sigma):
    gamma = []
    # Compute kernel
    for i in range(A.shape[0]):
        # K_A = rbf_kernel(A[:i+1],A[:i+1])
        K_A = matern_kernel(A[:i + 1], A[:i + 1])
        gamma.append(InfoGain(K_A,sigma))
    return gamma


def posterior_mean_cov_t(x,sigma,A_t,y_t):
    mu_t = np.zeros(x.shape[0])
    sigma_t = np.zeros(x.shape[0])

    A_t = np.array(A_t)
    y_t = np.array(y_t)

    # Reduce computation
    # K = rbf_kernel(A_t, A_t)
    K = matern_kernel(A_t, A_t)
    K = np.reshape(K,(K.shape[0],K.shape[0]))
    inv = np.linalg.pinv(K + np.identity(K.shape[0]) * sigma)
    for i in range(x.shape[0]):
        # k_t_vector = tbf_vector(A_t,x[i])
        k_t_vector = matern_kernel(A_t, x[i])
        if(A_t.shape[0] == 1):
            m = k_t_vector.T * inv * y_t
            # s = rbf_single(x[i], x[i]) - np.matmul(k_t_vector * inv, k_t_vector.T)
            s = matern_kernel(x[i], x[i]) - np.matmul(k_t_vector * inv, k_t_vector.T)
        else:
            m = np.matmul(np.matmul(k_t_vector.T, inv), y_t)
            # s = rbf_single(x[i],x[i]) - np.matmul(np.matmul(k_t_vector.T,inv),k_t_vector)
            s = matern_kernel(x[i], x[i]) - np.matmul(np.matmul(k_t_vector.T, inv), k_t_vector)
        mu_t[i] = m
        sigma_t[i] = s
    mu_t = np.array(mu_t)
    sigma_t = np.array(sigma_t)
    return mu_t,sigma_t

# Parameter B_t
def adaptive_B(t,D,delta):
    B = 2 * np.log(D*np.square((t+1)*np.pi)/ (6*delta))
    return B

def GP_UCB(D, sigma,T,delta):
    # Range of X
    x = np.linspace(0,5,D)
    # Sample f from RBF prior
    X_star = np.reshape(x,(-1,1))
    mu_p = np.zeros(X_star.shape[0])
    # sigma_p = rbf_kernel(X_star,X_star)
    sigma_p = matern_kernel(X_star, X_star)
    f_star = np.random.multivariate_normal(mu_p,sigma_p)
    # Find optimal decision
    f_optimum = np.max(f_star)

    R_t = []
    A_t = []
    y_t = []
    mu_t = np.zeros(x.shape[0])
    sigma_t = np.ones(x.shape[0])
    greedy_A = []
    B = []
    for t in range(T):
        B.append(adaptive_B(t,D,delta))
        values = mu_t + np.sqrt(B[t]*sigma_t)
        index = np.argmax(values)
        # index = np.random.randint(1000)

        # Equation 4
        greedy_A.append(x[np.argmax(sigma_t)])

        # Choose UCB X,Y
        f = f_star[index]
        y = f + np.random.normal(0,sigma)

        # Instantaneous Regret
        R_t.append(f_optimum-f)

        # Samples
        A_t.append(x[index])
        y_t.append(y)

        # Bayesian update
        mu_t, sigma_t = posterior_mean_cov_t(x, sigma, A_t, y_t)

        # if (t==1 ):
        #     plt.plot(values, label=t)

        if(t > 0):
            plt.plot(values)
            # plt.plot(mu_t)

    # Plot function sampled from prior
    plt.plot(f_star)
    # plt.legend()
    plt.show()

    # Greedy samples for maximal information
    greedy_A = np.array(greedy_A)
    gamma = greedy_information_gain(greedy_A,sigma)

    # Compute regret bound
    bound = np.arange(T)+1
    C = 8/np.log(1 + 1/sigma)
    print(C)
    B = np.array(B)
    print(B[-1])
    bound = bound * B
    bound *= C
    bound *= gamma
    print(gamma[-1])
    bound *= (np.e /(np.e -1))
    print (np.e /(np.e -1))
    bound = np.sqrt(bound)

    R_t = np.array(R_t)
    plt.title('Cumulative Regret')
    plt.ylabel('Sum')
    plt.xlabel('iterations')
    R_t = np.cumsum(R_t)
    plt.plot(R_t,label='Cumulative Regret')
    plt.plot(bound, label='Upper bound')
    plt.legend()
    plt.show()

    plt.plot(gamma, label='Gamma')
    plt.plot(2*np.square(np.log( np.arange(T)+1)), label='Upper bound for RBF')
    plt.legend()
    plt.show()

# sigma = 0.025
sigma = 1
D = 3000
delta = 0.1
T = 30
GP_UCB(D,sigma,T,delta)

def compute_mean_cov(X,X_star,f,K):
    inv = np.linalg.pinv(K(X, X))
    K_s = K(X_star, X)
    K_ss = K(X_star, X_star)
    K_s_to_inv = np.matmul(K_s, inv)

    mu = np.matmul(K_s_to_inv,f)
    sigma = K_ss - np.matmul(K_s_to_inv, K_s.T)

    return mu,sigma


def linear_kernel(X_p,X_q):
    K = np.matmul(X_p, X_q.T)
    return K



#
# X_star = np.linspace(-5,5)
# X_star = np.reshape(X_star,(-1,1))
#
# mu = np.zeros(50)
# sigma = rbf_kernel(X_star,X_star)
#
# f_star = np.random.multivariate_normal(mu,sigma,10)
# for i in range(10):
#     plt.plot(X_star,f_star[i])
# plt.title('Samples from Linear Prior')
# plt.show()
#
# X = np.array([-2,1,2,4])
# X = np.reshape(X,(-1,1))
# f = np.sin(X)
#
# mu,sigma = compute_mean_cov(X,X_star,f,rbf_kernel)
# mu = mu.reshape(-1)
# f_star = np.random.multivariate_normal(mu,sigma,10)
# for i in range(5):
#     plt.plot(X_star,f_star[i],color='tab:blue')
#
# plt.title('Samples from Linear Posterior')
# plt.plot(X_star,np.sin(X_star),label='Sine',color='tab:red')
# plt.scatter(X,f,s= 100,label='Observations',color = 'tab:green')
# plt.legend()
#
#
# plt.show()