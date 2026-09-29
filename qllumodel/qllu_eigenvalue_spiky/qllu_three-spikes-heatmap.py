# -*- coding: utf-8 -*-
"""
Plot heatmap of eigenvalue around spiky stationary solution of QLLU model
Number of spikes N = 3
2026-5-28: Created by Kensuke Ohtake
2026-9-17: Updated
2026-9-19: Final review completed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# set parameters
mu = 0.6 # mu
Lm = 1.0 # total manufacturing workers
Ph = 10.0 # total agricultural workers
r = 1.0 # radius
F = 1.0 # fixed input
v = 1.0 # adjustment speed
lm = Lm / 3.0 # statinary mobile population (weight of delta function)
ph = Ph / (2.0 * np.pi * r) # homogeneous immobile population (density)

# eigenvalue function
def eigenv(t, s):

    # t: tau
    # s: sigma
    
    alp = t * (s - 1.0) # alpha
    ep = np.exp(alp * (2.0 / 3.0) * np.pi * r) # exp_plus
    em = np.exp(-alp * (2.0 / 3.0) * np.pi * r) # exp_minus
    if Ph <= (s * Lm * alp * np.pi * r) / (1.0 - em):
        print('warning: sufficient condition for positive agriculture')
    GP = (3.0 * F) / (Lm * (1.0 + 2.0 * em)) # stationary priceindex raised to the power of sigma-1
    
    # Matrices
    Ex = np.array([[1.0, em, em],
                   [em, 1.0, em],
                   [em, em, 1.0]]) # transport matrix
    e11 = 1.0 + 2.0 * np.power(em, 2.0)
    e12 = 2.0 * em + np.power(em, 2.0)
    ExEx = np.array([[e11, e12, e12],
                     [e12, e11, e12],
                     [e12, e12, e11]])
    intexp_11 = (
        2.0 * np.pi * r / 3.0
        + (1.0 / alp) * ((1.0 / np.power(1.0 + em, 2.0)) 
                         + (1.0 / np.power(1.0 + ep, 2.0)) - 1.0)
        * np.log((2.0 + ep) / (2.0 + em))
        + (1.0 / alp) * ((1.0 / np.power(1.0 + em, 2.0)) 
                         + (1.0 / np.power(1.0 + ep, 2.0)) + 1.0)
        * ((em - ep) / ((2.0 + em) * (2.0 + ep)))
        )
    intexp_12 = (
        (1.0 / (alp * (1.0 + em) * (1.0 + ep)))
        * np.log((2.0 + ep) / (2.0 + em))
        - ((1.0 + em + ep) / (alp * (1.0 + em) * (1.0 + ep)))
        * ((em - ep) / ((2.0 + em) * (2.0 + ep)))
        )
    intexp = np.array([[intexp_11, intexp_12, intexp_12],
                       [intexp_12, intexp_11, intexp_12],
                       [intexp_12, intexp_12, intexp_11]])
    
    A1 = (mu / (s * F)) * GP * Ex
    A2 = -(mu / (s * np.power(F, 2.0))) * lm * np.power(GP, 2.0) * ExEx
    A3 = -(mu / s) * (ph / np.power(lm, 2.0)) * intexp
    A = A1 + A2 + A3
    
    L = v * lm * (A + (mu / (F * (s - 1.0))) * GP * Ex)

    val, vec = np.linalg.eigh(L)
    
    return val, L # eigenvalues and linerized operator

# generate meshgrid
tau_size = 255
sigma_size = 255
tau_space = np.linspace(0.01, 1.5, tau_size)
sigma_space = np.linspace(1.1, 5.5, sigma_size)
tau_mesh, sigma_mesh = np.meshgrid(tau_space, sigma_space)

# compute maximal eigenvalues
maxevs = np.zeros((sigma_size, tau_size))
i = 0
for sig in sigma_space:
    j = 0
    for tau in tau_space:
        val, L = eigenv(tau, sig)
        if np.allclose(L, L.T) == False:
            print('Error: not symmetric matrix!')
        maxevs[i, j] = np.max(val)
        j += 1
    i += 1

# plot heatmap
fig, ax = plt.subplots()
hm_norm = mcolors.TwoSlopeNorm(vmin=maxevs.min(), vcenter=0, vmax=maxevs.max())
hm_levels = np.linspace(maxevs.min(), maxevs.max(), 256)
cont = ax.contour(tau_mesh, sigma_mesh, maxevs, levels=[0], linewidths=1.5)
contf = ax.contourf(tau_mesh, sigma_mesh, maxevs, levels=hm_levels, cmap='jet', norm=hm_norm)
#cmap='rainbow' 'bwr' 'coolwarm' 'seismic'    
ticks = np.linspace(maxevs.min(), maxevs.max(), 9)
fig.colorbar(contf, extend='max').set_ticks(ticks)

ax.set_aspect('auto', adjustable='box')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma$')
plt.savefig('QLLU_3spikes_heatmap.png', format='png', dpi=300)
plt.show()
