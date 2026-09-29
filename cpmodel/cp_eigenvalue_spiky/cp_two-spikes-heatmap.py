# -*- coding: utf-8 -*-
"""
Plot heatmap of eigenvalue around spiky stationary solution of CP model
Number of spikes N = 2
2026-5-23: Created by Kensuke Ohtake
2026-9-19: Updated
2026-9-19: Final review completed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# set parameters
mu = 0.6 # mu
r = 1.0 # radius
v = 1.0 # adjustment speed
lm = 1.0 / 2.0 # statinary mobile population (weight of delta function)
ph = 1.0 / (2.0 * np.pi * r) # homogeneous immobile population (density)

# eigenvalue function
def eigenv(t, s):

    # t: tau
    # s: sigma
    
    alp = t * (s - 1.0) # alpha
    ep = np.exp(alp * np.pi * r) # exp_plus
    em = np.exp(-alp * np.pi * r) # exp_minus
    w = 1.0 # stationary nominal wage
    G = np.power((1.0 + em) / 2.0, 1.0 / (1.0 - s)) # stationary priceindex
    GP = 2.0 / (1.0 + em) # stationary priceindex raised to the power of sigma-1
    
    # Matrices
    I = np.eye(2, dtype=float) # identity matrix
    Ex = np.array([[1.0, em],
                   [em, 1.0]]) # transport matrix
    ExEx = np.array([[1.0 + np.power(em, 2.0), 2.0 * em],
                     [2.0 * em, 1.0 + np.power(em, 2.0)]])
    
    intexp_11 = (
        (1.0 / alp) * (np.log((1.0 + ep) / (1.0 + em))
           + ((em - ep) / ((1.0 + em) * (1.0 + ep))))
           )
    intexp_12 = (
        -(1.0 / alp) * ((em - ep) / ((1.0 + em) * (1.0 + ep)))
        )
    intexp = np.array([[intexp_11, intexp_12],
                       [intexp_12, intexp_11]])
    A = (
        ((mu * lm * GP * np.power(w, 1.0 - s)) / s) * Ex
        + ((mu * (s - 1.0) * np.power(lm, 2.0) * np.power(w, 2.0 - 2.0 * s) * np.power(GP, 2.0)) / s) * ExEx
        + (((1.0 - mu) * (s - 1.0) * ph) / (s * lm * w)) * intexp
        )
    B = (
        ((mu * GP * np.power(w, 2.0 - s)) / s) * Ex
        - ((mu * lm * np.power(w, 3.0 - 2.0 * s) * np.power(GP, 2.0)) / s) * ExEx
        - (((1.0 - mu) * ph) / (s * np.power(lm, 2.0))) * intexp
        )
    C = lm * np.power(w, -s) * np.power(G, s) * Ex
    D = (1.0 / (1.0 - s)) * np.power(w, 1.0 - s) * np.power(G, s) * Ex
    
    dw = np.dot(np.linalg.inv(I - A), B) # Frechet derivative of w
    dG = np.dot(C, dw) + D # Frechet derivative of G
    
    L = v * lm * np.power(G, - mu) * (dw - (mu * w / G) * dG)

    val, vec = np.linalg.eigh(L)
    
    return val, L # eigenvalues and linerized operator

# generate meshgrid
tau_size = 255
sigma_size = 255
tau_space = np.linspace(0.01, 15.0, tau_size)
sigma_space = np.linspace(2.5, 7.0, sigma_size)
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
plt.savefig('CP_2spikes_heatmap.png', format='png', dpi=300)
plt.show()
