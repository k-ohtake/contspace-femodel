# -*- coding: utf-8 -*-
"""
Plot heatmap of eigenvalue around spiky stationary solution of FE model
Number of spikes N = 2
2026-3-12: Created by Kensuke Ohtake
2026-9-13: Updated
2026-9-13: Final review completed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# set parameters
mu = 0.6 # mu
Lm = 1.0 # total manufacturing workers
Ph = 1.0 # total agricultural workers
r = 1.0 # radius
F = 1.0 # fixed input
v = 1.0 # adjustment speed
lm = Lm / 2.0 # statinary mobile population (weight of delta)
ph = Ph / (2.0 * np.pi * r) # homogeneous immobile population (density)

# eigenvalue function
def eigenv(t, s):

    # t: tau
    # s: sigma
    
    alp = t * (s - 1.0) # alpha
    ep = np.exp(alp * np.pi * r) # exp_plus
    em = np.exp(-alp * np.pi * r) # exp_minus
    w = (mu / (s - mu)) * (Ph / Lm) # stationary nominal wawage
    G = np.power((Lm / (2.0 * F)) * (1.0 + em), 1.0 / (1.0 - s)) # stationary priceindex
    
    # matrices
    I = np.eye(2, dtype=float) # identity matrix
    Ex = np.array([[1.0, em],
                   [em, 1.0]]) # transport matrix
    ExEx = np.array([[1.0 + np.power(em, 2.0), 2.0 * em],
                     [2.0 * em, 1.0 + np.power(em, 2.0)]])
    
    A = (mu / (s * F)) * lm * np.power(G, s - 1.0) * Ex
    B1 = (mu / (s * F)) * w * np.power(G, s - 1.0) * Ex
    B2 = -(mu / (s * np.power(F, 2.0))) * lm * w * np.power(G, 2.0 * s - 2.0) * ExEx
    intexp_11 = (
        (1.0 / alp) * (np.log((1.0 + ep) / (1.0 + em))
           + ((em - ep) / ((1.0 + em) * (1.0 + ep))))
           )
    intexp_12 = (
        (1.0 / alp) * ((ep - em) / ((1.0 + em) * (1.0 + ep)))
        )
    intexp = np.array([[intexp_11, intexp_12],
                       [intexp_12, intexp_11]])
    B3 = -(mu / s) * (ph / np.power(lm, 2.0)) * intexp
    B = B1 + B2 + B3
    
    dG = (np.power(G, s) / (F * (1.0 - s))) * Ex # Frechet der. of G
    dw = np.dot(np.linalg.inv(I - A), B) # Frechet der. of w
    L = v * lm * np.power(G, - mu) * (dw - mu * (w / G) * dG) # linerized operator

    val, vec = np.linalg.eigh(L)
    
    return val, L # eigenvalues and linerized operator

# generate meshgrid
tau_size = 255
sigma_size = 500
tau_space = np.linspace(0.01, 15.0, tau_size)
sigma_space = np.linspace(1.6, 7.0, sigma_size)
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
plt.savefig('FE_2spikes_heatmap.png', format='png', dpi=300)
plt.show()
