# -*- coding: utf-8 -*-
"""
Plot eigenvalue around homogeneous stationary solution of FE model
2025-4-10: Created by Kensuke Ohtake
2026-9-21: Updated
2026-9-21: Final review completed
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

# get datetime
DateTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

# set parameters
mu = 0.6 # mu
Lam = 1.0 # Lamdba: total manufacturing workers
Phi = 1.0 # Phi: total agricultural workers
r = 1.0 # radius
F = 1.0 # fixed input
v = 1.0 # migration coefficient
lam = Lam / (2.0 * np.pi * r) # lambda: homogeneous mobile population
phi = Phi / (2.0 * np.pi * r) # phi: homogeneous immobile population

# define eigenvalue function
def eigenvalue(tau, sig, k):

    # tau: tau vector
    # sig: sigma
    # k: frequency number
    
    # alpha vector
    alp = (sig - 1.0) * tau
    
    # homogeneous nominal wage and price index
    w = ((mu * phi) / (sig * lam)) / (1.0 - (mu / sig)) # nomilal wage
    G = np.power(2.0 * lam * ((1.0 - np.exp(-alp * np.pi * r)) / (F * alp)), 1.0 / (1.0 - sig)) # price index
    
    # two terms to compute Z
    quad_term = (np.power(alp, 2.0) * np.power(r, 2.0)) / (np.power(k, 2.0) + np.power(alp, 2.0) * np.power(r, 2.0))
    exp_term = (1.0 + np.exp(-alp * r * np.pi)) / (1.0 - np.exp(-alp * r * np.pi))
    
    if k % 2 == 0:# when k is even number
        Z = quad_term
    else:# when k is odd number
        Z = quad_term * exp_term

    Gamma_k = -v * mu * np.power(G, -mu) * ((w / (1.0 - sig)) + (((w + (phi / lam)) * Z - w) / (sig - mu * Z))) * Z
    
    return Gamma_k

# set coordinate
t = np.linspace(0.01, 3.0, 256) # tau vector
s = 3.0 # sigma

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
plt.ylim(-0.01, 0.015)
plt.axhline(y=0, color='gray', linestyle='dotted')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.legend()
plt.savefig('FE_eigenvalues.png', format='png', dpi=300)
plt.show()
