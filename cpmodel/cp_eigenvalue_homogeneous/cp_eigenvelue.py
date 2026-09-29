# -*- coding: utf-8 -*-
"""
Plot eigenvalue around homogeneous stationary solution of CP model
2025-3-6: Created by Kensuke Ohtake
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
r = 1.0 # radius
v = 1.0 # migration coefficient
lam = 1.0 / (2.0 * np.pi * r) # lambda: homogeneous manufactuirng population
phi = 1.0 / (2.0 * np.pi * r) # phi: homogeneous agricultural population

# define eigenvalue function
def eigenvalue(tau, sig, k):

    # tau: tau vector
    # sig: sigma
    # k: frequency number
    
    # alpha vector
    alp = (sig - 1.0) * tau
    
    # homogeneous price index
    G = np.power((1.0 - np.exp(-alp * np.pi * r)) / (alp * np.pi * r), 1.0 / (1.0 - sig))
    
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
    first_term = (1.0 - mu * Z) * (((mu / sig) * Z - (1.0 / sig) * np.power(Z, 2.0)) / (1.0 - (mu / sig) * Z - ((sig - 1.0) / sig) * np.power(Z, 2.0)))
    second_term = (mu * Z) / (sig - 1.0)
    Gamma_k = v * np.power(G, -mu) * (first_term + second_term)
    return Gamma_k

# set coordinate
t = np.linspace(0.01, 10.0, 256) # tau vector
s = 5.0 # sigma

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
plt.ylim(-0.15, 0.2)
plt.axhline(y=0, color='gray', linestyle='dotted')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.savefig('CP_eigenvalues.png', format='png', dpi=300)
plt.legend()
plt.show()
