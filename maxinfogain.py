import numpy as np

def aux_data(dim, numrows):
    mu, sigma = 0, 1
    mean = np.random.normal(mu, sigma, dim)
    X = np.random.multivariate_normal(mean, np.eye(dim), numrows)
    
    return X

def Gaussian_noise(size):
    mean = np.zeros(size)
    sigma = np.random.normal(0, 1)
    cov = (sigma**2)*np.eye(size)
    noise = np.random.multivariate_normal(mean, cov)
    
    return sigma, noise

def Gauss_kernel(A, sigma, lengthscale):
    '''
    RBF
    Comment: need to ensure k(x, x') <= 1
    '''
    K = np.zeros(( len(A), len(A) ))
    for i, x1 in enumerate(A):
        for j, x2 in enumerate(A):
            numerator = (np.linalg.norm(np.subtract(x1, x2)))**2
            denom = 2*(lengthscale**2)
            K[i][j] = (sigma**2)*np.exp(-numerator/denom)
    return K

def InfoGain(A_dim, kernel_A, sigma_noise): 
    '''
    I(y_A; f_A) = 1/2 * log * det( I + sigma**(-2)*K_A )
    K_A = k(x, x'), x, x' in A
    '''
    
    return 0.5 * np.log(np.add(np.eye(len(kernel_A)), (sigma_noise**2)*kernel_A))

def MaxInfoGain(T, dimension):
    dimension = 8
    inf_gain = []
    for t in range(1, T+1):
        A = aux_data(dimension, T) 
        K = Gauss_kernel(A, sigma=1, lengthscale=1) 
        sigma, _ = Gaussian_noise(dimension)
        inf_gain.append(InfoGain(dimension, K, sigma))
        
    return np.max(inf_gain)

def main():
        T = 10
        dim = 7
        gamma_t = MaxInfoGain(T, dim)
        print(gamma_t)

if __name__ == "__main__":
    main()
