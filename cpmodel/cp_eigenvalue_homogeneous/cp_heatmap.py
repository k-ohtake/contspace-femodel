# -*- coding: utf-8 -*-
"""
Plot heatmap of eigenvalue around homogeneous stationary solution of CP model
2025.3.5: Created by Kensuke Ohtake
2026-9-20: Updated
2026-9-20: Final review completed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

    # tau: tau mash
    # sig: sigma mesh
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

# set parameter space
tau_coordinate = np.linspace(0.01, 2.5, 550) # tau-coordinate
sigma_coordinate = np.linspace(1.1, 17.0, 1025) # sigma-coordinate
tau_mesh, sigma_mesh = np.meshgrid(tau_coordinate, sigma_coordinate) # make meshgrid

# compute and plot eigenvalues
frs = [1, 2, 3, 4, 5, 6] # frequency numbers
for n in frs:
    fig, ax = plt.subplots()
    ev = eigenvalue(tau_mesh, sigma_mesh, n)
    hm_norm = mcolors.TwoSlopeNorm(vmin=ev.min(), vcenter=0, vmax=ev.max())
    hm_levels = np.linspace(ev.min(), ev.max(), 256)
    cont = ax.contour(tau_mesh, sigma_mesh, ev, levels=[0], linewidths=1.5)
    contf = ax.contourf(tau_mesh, sigma_mesh, ev, levels=hm_levels, cmap='jet', norm=hm_norm)
    #cmap='rainbow' 'bwr' 'coolwarm' 'seismic'    
    ticks = np.linspace(ev.min(), ev.max(), 9)
    fig.colorbar(contf, extend='max').set_ticks(ticks)
    
    ax.set_aspect('auto', adjustable='box')
    ax.set_title(r'k={}'.format(n))

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\sigma$')
    plt.savefig('CP_heatmap_k_{}.png'.format(n), format='png', dpi=300)
    plt.show()
    pass
