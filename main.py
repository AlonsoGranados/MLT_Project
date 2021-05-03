import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern


def linear_kernel(X_p,X_q):
    X_p = np.reshape(X_p, (-1, 1))
    X_q = np.reshape(X_q, (-1, 1))
    K = np.matmul(X_p, X_q.T)
    return K


def matern_kernel(X_p,X_q):
    X_p = np.reshape(X_p, (-1, 1))
    X_q = np.reshape(X_q, (-1, 1))
    nu = 2.5
    l = 1
    k = Matern(length_scale=l ,nu = nu)
    K = k.__call__(X_p,X_q)
    return K


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
def greedy_information_gain(A,sigma,k):
    gamma = []
    # Compute kernel
    for i in range(A.shape[0]):
        if (k == 1):
            K_A = rbf_kernel(A[:i+1],A[:i+1])
        elif (k == 2):
            K_A = matern_kernel(A[:i + 1], A[:i + 1])
        else:
            K_A = linear_kernel(A[:i + 1], A[:i + 1])
        gamma.append(InfoGain(K_A,sigma))
    return gamma


def posterior_mean_cov_t(x,sigma,A_t,y_t,k):
    mu_t = np.zeros(x.shape[0])
    sigma_t = np.zeros(x.shape[0])

    A_t = np.array(A_t)
    y_t = np.array(y_t)

    # Reduce computation
    if(k == 1):
        K = rbf_kernel(A_t, A_t)
    elif(k==2):
        K = matern_kernel(A_t, A_t)
    else:
        K = linear_kernel(A_t, A_t)
    K = np.reshape(K,(K.shape[0],K.shape[0]))
    inv = np.linalg.pinv(K + np.identity(K.shape[0]) * sigma)
    for i in range(x.shape[0]):
        if(k==1):
            k_t_vector = tbf_vector(A_t,x[i])
        elif(k==2):
            k_t_vector = matern_kernel(A_t, x[i])
        else:
            k_t_vector = linear_kernel(A_t, x[i])
        if(A_t.shape[0] == 1):
            m = k_t_vector.T * inv * y_t
            if (k == 1):
                s = rbf_single(x[i], x[i]) - np.matmul(k_t_vector * inv, k_t_vector.T)
            elif (k == 2):
                s = matern_kernel(x[i], x[i]) - np.matmul(k_t_vector * inv, k_t_vector.T)
            else:
                s = linear_kernel(x[i], x[i]) - np.matmul(k_t_vector * inv, k_t_vector.T)
        else:
            m = np.matmul(np.matmul(k_t_vector.T, inv), y_t)
            if (k == 1):
                s = rbf_single(x[i],x[i]) - np.matmul(np.matmul(k_t_vector.T,inv),k_t_vector)
            elif(k == 2):
                s = matern_kernel(x[i], x[i]) - np.matmul(np.matmul(k_t_vector.T, inv), k_t_vector)
            else:
                s = linear_kernel(x[i], x[i]) - np.matmul(np.matmul(k_t_vector.T, inv), k_t_vector)
        mu_t[i] = m
        sigma_t[i] = s
    mu_t = np.array(mu_t)
    sigma_t = np.array(sigma_t)
    return mu_t,sigma_t

# Parameter B_t
def adaptive_B(t,D,delta):
    B = 2 * np.log(D*np.square((t+1)*np.pi)/ (6*delta))
    return B

def GP_UCB(D, sigma,T,delta,k):
    # Range of X
    x = np.linspace(0,5,D)
    # Sample f from RBF prior
    X_star = np.reshape(x,(-1,1))
    mu_p = np.zeros(X_star.shape[0])
    if(k == 1):
        sigma_p = rbf_kernel(X_star,X_star)
    elif(k == 2):
        sigma_p = matern_kernel(X_star, X_star)
    else:
        sigma_p = linear_kernel(X_star, X_star)
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
        mu_t, sigma_t = posterior_mean_cov_t(x, sigma, A_t, y_t,k)

        if(t == 9 or t == 19 or t == 29):
            # plt.plot(values)
            plt.plot(x,mu_t,label = 'Predicted mean step {0}'.format(t+1))
            plt.fill_between(x, mu_t + np.sqrt(B[t]*sigma_t), mu_t,color = 'tab:gray')
            # Plot function sampled from prior
            plt.plot(x, f_star, label='True function')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()

    # # Plot function sampled from prior
    # plt.plot(x,f_star,label = 'True function')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()

    # Greedy samples for maximal information
    greedy_A = np.array(greedy_A)
    gamma = greedy_information_gain(greedy_A,sigma,k)

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
    if (k == 1):
        plt.title('Cumulative Regret for RBF kernel')
    elif (k == 2):
        plt.title('Cumulative Regret for Matern Kernel')
    else:
        plt.title('Cumulative Regret for Linear Kernel')
    plt.ylabel('Regret')
    plt.xlabel('iterations')
    R_t = np.cumsum(R_t)
    plt.plot(R_t,label='Cumulative Regret')
    plt.plot(bound, label='Regret bound')
    plt.legend()
    plt.show()

    plt.xlabel('iterations')
    plt.xlabel('Information Gain')
    if(k==1):
        plt.title('RBF Kernel Information Gain')
        plt.plot(gamma, label='Empirical IG')
        plt.plot(np.square(np.log( np.arange(T)+1)), label='O(log(T)^d+1) ')
        plt.legend()
        plt.show()
    elif (k == 2):
        nu = 2.5
        plt.title('Matern Kernel Information Gain')
        plt.plot(gamma, label='Empirical IG')
        plt.plot(np.power(np.arange(T)+1,2/(2+2*nu))*np.log(np.arange(T)+1), label='O(T^(d(d+1)/2v+d(d+1))log(T))')
        plt.legend()
        plt.show()
    else:
        plt.title('Linear Kernel Information Gain')
        plt.plot(gamma, label='Empirical IG')
        plt.plot(np.log(np.arange(T) + 1), label='O(d log(T)) ')
        plt.legend()
        plt.show()

# sigma = 0.025
sigma = 0.1
D = 3000
delta = 0.25
T = 30
GP_UCB(D,sigma,T,delta,3)

# # Draw samples from prior
# # Range of X
# x = np.linspace(0, 5, D)
# # Sample f from RBF prior
# X_star = np.reshape(x, (-1, 1))
# mu_p = np.zeros(X_star.shape[0])
# # sigma_p = rbf_kernel(X_star,X_star)
# sigma_p = matern_kernel(X_star, X_star)
# # sigma_p = linear_kernel(X_star, X_star)
# f_star = np.random.multivariate_normal(mu_p, sigma_p,10)
# print(f_star.shape)
# for i in range(f_star.shape[0]):
#     plt.plot(x,f_star[i,:])
# plt.title('Samples from Matern kernel')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.show()
