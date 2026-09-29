# -*- coding: utf-8 -*-
"""
Plot eigenvalue around homogeneous stationary solution of QLLU model
2025-3-5: Created by Kensuke Ohtake
2026-9-20: Updated
2026-9-20: Final review completed
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

# get datetime
DateTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

# set parameters
mu = 0.6 # mu
Lam = 1.0 # Lambda: total manufacturing workers
Phi = 10.0 # Phi: total agricultural workers
r = 1.0 # radius
v = 1.0 # adjustment speed
lam = Lam / (2.0 * np.pi * r) # lambda: homogeneous manufactuirng population
phi = Phi / (2.0 * np.pi * r) # phi: homogeneous agricultural population

# define eigenvalue function
def eigenvalue(tau, sig, k):

    # tau: tau vector
    # sig: sigma
    # k: frequency number
    
    # alpha vector
    alp = (sig - 1.0) * tau
    
    # convert k to real number
    k = float(k)
    
    # two terms to compute Z
    quad_term = (np.power(alp, 2.0) * np.power(r, 2.0)) / (np.power(k, 2.0) + np.power(alp, 2.0) * np.power(r, 2.0))
    exp_term = (1.0 + np.exp(-alp * r * np.pi)) / (1.0 - np.exp(-alp * r * np.pi))
    
    if k % 2 == 0:# when k is even number
        Z = quad_term
    else:# when k is odd number
        Z = quad_term * exp_term
    
    # compute engenvalue for frequency number k                                                                                                                                           
    Gamma_k = v * (mu / sig) * (-((lam + phi) / lam) * np.power(Z, 2.0) + ((2.0 * sig - 1.0) / (sig - 1.0)) * Z)
    
    return Gamma_k

# set coordinate
t = np.linspace(0.01, 3, 256) # tau vector
s = 2.0 # sigma

# compute and plot eigenvalues
frs = [1, 2, 3, 4, 5, 6] # frequency numbers
fig, ax = plt.subplots()
for n in frs:
    ev = eigenvalue(t, s, n)
    plt.plot(t, ev, label='k={}'.format(n))
    pass

# plot formatting
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Gamma_k$')
plt.ylim(-0.05, 0.15)
plt.axhline(y=0, color='gray', linestyle='dotted')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.savefig('QLLU_eigenvalues.png', format='png', dpi=300)
plt.legend()
plt.show()
