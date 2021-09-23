import pandas as pd
import numpy as np

from pyeb.utils import inv_logit

def sim_binary_data(n_obs = 5000, n_dim = 10, seed = None):
    if seed is not None:
        np.random.seed(seed)
        
    X = np.random.normal(size = (n_obs, n_dim))
    XD = np.hstack([np.ones((n_obs,1)), X])
    betas = np.random.normal(size = n_dim + 1)
    ps = inv_logit(XD.dot(betas))
    A = np.random.binomial(1, ps)
    AX = np.hstack([A.reshape((n_obs, 1)), X])
    cnames = ['TA']
    cnames.extend(['COV' + str(i) for i in np.arange(1,n_dim+1)])
    Xdf = pd.DataFrame(AX, columns=cnames)
    return Xdf