import numpy as np

def inv_logit(x):
    return 1. / (1. + np.exp(-x))

def w_mean(w, x):
    return x.dot(w) / np.sum(w)

def w_var(w, x):
    z = (x - w_mean(w,x))**2
    return z.dot(w) / np.sum(w)

def std_diff_means(x, ta, w=None):
    if w is None:
        w = np.ones(x.shape[0])
    bar_xt = w_mean(w[ta==1], x[ta == 1])
    bar_xc = w_mean(w[ta==0], x[ta == 0])
    var_xt = w_var(w[ta==1], x[ta == 1])
    var_xc = w_var(w[ta==0], x[ta == 0])
    sdm = (bar_xt - bar_xc) / np.sqrt( (var_xt + var_xc) / 2. )
    return sdm

def log_ratio_sd(x, ta, w=None):
    if w is None:
        w = np.ones(x.shape[0])
    sd_xt = np.sqrt(w_var(w[ta==1], x[ta == 1]))
    sd_xc = np.sqrt(w_var(w[ta==0], x[ta == 0]))
    return np.log(sd_xt) - np.log(sd_xc)

def ess(w):
    return np.sum(w)**2 / np.sum(w**2)