import numpy as np

if __name__ == "__main__":
    mu = np.array([0,0])
    sigma = np.array([[1,0],[0,1]])
    sample = np.random.multivariate_normal(mu,sigma)
    print(sample)