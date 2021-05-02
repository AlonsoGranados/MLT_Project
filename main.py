import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
from scipy.special import gamma
from sklearn.gaussian_process.kernels import Matern


def rbf_kernel(X_p,X_q):
    X_p = np.reshape(X_p,(-1,1))
    X_q = np.reshape(X_q, (-1, 1))
    K = -2 * np.matmul(X_p,X_q.T)
    K += X_p * X_p
    K += X_q.T * X_q.T
    K = K * (-1/2)
    K = np.exp(K)
    return K

def tbf_vector(A_t,x):
    v = np.zeros((A_t.shape[0],1))
    for i in range(v.shape[0]):
        v[i] = rbf_single(A_t[i],x)
    return v

def rbf_single(x,y):
    d = np.square(x - y)
    d = d * -1/2
    d = np.exp(d)
    return d


def InfoGain(kernel_A, sigma_noise):
    '''
    I(y_A; f_A) = 1/2 * log * det( I + sigma**(-2)*K_A )
    K_A = k(x, x'), x, x' in A
    '''

    return 0.5 * np.log(np.linalg.det(np.add(np.eye(len(kernel_A)), (sigma_noise ** -1) * kernel_A)))


def greedy_information_gain(A,sigma):
    gamma = []
    # Compute kernel
    for i in range(A.shape[0]):
        K_A = rbf_kernel(A[:i+1],A[:i+1])
        gamma.append(InfoGain(K_A,sigma))
    return gamma


def posterior_mean_cov_t(x,sigma,A_t,y_t):
    mu_t = np.zeros(x.shape[0])
    sigma_t = np.zeros(x.shape[0])

    A_t = np.array(A_t)
    y_t = np.array(y_t)

    # Save computation
    K = rbf_kernel(A_t, A_t)
    K = np.reshape(K,(K.shape[0],K.shape[0]))
    #
    inv = np.linalg.pinv(K + np.identity(K.shape[0]) * sigma)
    for i in range(x.shape[0]):
        k_t_vector = tbf_vector(A_t,x[i])
        if(A_t.shape[0] == 1):
            m = k_t_vector.T * inv * y_t
            s = rbf_single(x[i], x[i]) - np.matmul(k_t_vector * inv, k_t_vector.T)
        else:
            m = np.matmul(np.matmul(k_t_vector.T, inv), y_t)
            s = rbf_single(x[i],x[i]) - np.matmul(np.matmul(k_t_vector.T,inv),k_t_vector)
        mu_t[i] = m
        sigma_t[i] = s
    mu_t = np.array(mu_t)
    sigma_t = np.array(sigma_t)
    return mu_t,sigma_t

def adaptive_B(t,D,delta):
    B = 2 * np.log(D*np.square((t+1)*np.pi)/ (6*delta))
    return B

def GP_UCB(D, sigma,k,T,delta):
    x = np.linspace(0,D*5,3000)

    X_star = np.reshape(x,(-1,1))
    mu_p = np.zeros(X_star.shape[0])
    sigma_p = rbf_kernel(X_star,X_star)
    #
    f_star = np.random.multivariate_normal(mu_p,sigma_p)
    f_optimum = np.max(f_star)
    # print(f_star.shape)
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

        greedy_A.append(x[np.argmax(sigma_t)])
        print(index)
        # f = np.random.normal(mu_t[index], sigma_t[index])
        f = f_star[index]
        # print(f)
        # f = np.sin(x[index])
        y = f + np.random.normal(0,sigma)
        R_t.append(f_optimum-f)

        A_t.append(x[index])
        y_t.append(y)

        # Bayessian update
        mu_t, sigma_t = posterior_mean_cov_t(x, sigma, A_t, y_t)

        # if (t==1 ):
        #     plt.plot(values, label=t)

        if(t > 0):
            plt.plot(values)
            # plt.plot(mu_t)

        # print(mu_t.shape)
        # print(sigma_t.shape)
    plt.plot(f_star)
    plt.legend()
    plt.show()

    greedy_A = np.array(greedy_A)
    gamma = greedy_information_gain(greedy_A,sigma)
    # print(gamma)

    bound = np.arange(T)+1
    C = 8/np.log(1 + 1/sigma)
    B = np.array(B)
    # gamma = 1
    bound = bound * B
    bound *= C
    bound *= gamma
    print(np.e /(np.e -1))
    bound *= (np.e /(np.e -1))
    bound = np.sqrt(bound)
    # print(bound)

    R_t = np.array(R_t)
    plt.title('Cumulative Regret')
    plt.ylabel('Sum')
    plt.xlabel('iterations')
    R_t = np.cumsum(R_t)
    plt.plot(R_t,label='Cumulative Regret')
    plt.plot(bound, label='Upper bound')
    plt.legend()
    plt.show()

sigma = 0.025
# sigma = 1
D = 1
delta = 0.05
GP_UCB(D,sigma,rbf_kernel,30,delta)

def compute_mean_cov(X,X_star,f,K):
    inv = np.linalg.pinv(K(X, X))
    K_s = K(X_star, X)
    K_ss = K(X_star, X_star)
    K_s_to_inv = np.matmul(K_s, inv)

    mu = np.matmul(K_s_to_inv,f)
    sigma = K_ss - np.matmul(K_s_to_inv, K_s.T)

    return mu,sigma


def matern_kernel(X_p,X_q):
    nu = 2.5
    l = 1
    # K = -2 * np.matmul(X_p,X_q.T)
    # K += X_p * X_p
    # K += X_q.T * X_q.T
    # # distance
    # d = np.sqrt(K)
    # r = d * (np.sqrt(2 *nu)/l)
    # K = np.power(r,nu)
    # K *= kv(nu, r)
    # K = K /gamma(nu)
    # K = K / np.power(2,nu-1)
    # K[np.isnan(K)] = 0
    k = Matern(length_scale=l ,nu = nu)
    K = k.__call__(X_p,X_q)
    return K

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