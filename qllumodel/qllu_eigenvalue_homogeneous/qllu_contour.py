# -*- coding: utf-8 -*-
"""
Plot contour lines of eigenvalue around homogeneous stationary solution of QLLU model
2025-3-5: Created by Kensuke Ohtake
2026-9-20: Updated
2026-9-20: Final review completed
"""

import numpy as np
import matplotlib
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

    # t: tau mesh
    # s: sigma mesh
    # k: frequency number
    
    # alpha mesh
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

# set parameter space
tau_coordinate = np.linspace(0.01, 2.5, 550) # tau-coordinate
sigma_coordinate = np.linspace(1.1, 17.0, 1025) # sigma-coordinate
tau_mesh, sigma_mesh = np.meshgrid(tau_coordinate, sigma_coordinate) # make meshgrid

# compute and plot eigenvalues
labels = [] # list of labels
hs = [] # list of legend handles
fig, ax = plt.subplots()
frs = [1, 2, 3, 4, 5, 6] # frequency numbers
i = 0
for n in frs:
    ev = eigenvalue(tau_mesh, sigma_mesh, n)
    cont = ax.contour(tau_mesh, sigma_mesh, ev, levels=[0], linewidths=1.5, colors=[matplotlib.cm.tab10(i)])
    lab = r'$k$ = {}'.format(n)
    labels.append(lab)
    h, _ = cont.legend_elements() # h: legend handles generated from contour line
    hs.append(h[0]) # store the first legend handle
    i += 1
    pass

ax.legend(hs, labels) # Add legend consisting of legend handles and their corresponding labels
plt.gca().set_aspect('equal', adjustable='box')
ax.set_facecolor('0.95')
ax.set_aspect('auto', adjustable='box')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma$')
plt.grid(linestyle=':')
plt.savefig('QLLU_contours.png', format='png', dpi=300)
plt.show()
