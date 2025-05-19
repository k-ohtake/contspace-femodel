# -*- coding: utf-8 -*-
"""
2025.4.10 thu. created by Kensuke Ohtake
Footloose-Entrepreneur model
Eigenvalues analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime

# get datetime
DateTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

# set parameters
m = 0.6 # mu
Lam = 1.0 # total manufacturing workers
Ph = 1.0 # total agricultural workers
r = 1.0 # radius
F = 1.0 # fixed input
v = 1.0 # adjustment speed
lam = Lam/(2.0*np.pi*r) # homogeneous mobile population
ph = Ph/(2.0*np.pi*r) # homogeneous immobile population

def eigenv(X, Y, k):

    # X: tau mesh
    # Y: sigma mesh
    # k: frequency number (k)
    
    k = float(k)

    alp = (Y - 1.0) * X # alpha vector
    w = ((m * ph) / (sig * lam)) / (1.0 - (m / sig)) # homogeneous nomilal wage
    G = np.power(2.0 * lam * ((1.0 - np.exp(-alp * np.pi * r)) / (F * alp)), 1.0 / (1.0 - Y)) # homogeneous price index
    
    if k % 2 == 0:# when k is even number
        Z = (np.power(alp, 2.0) * np.power(r, 2.0)) / (np.power(k, 2.0) + np.power(alp, 2.0) * np.power(r, 2.0))
    else:# when k is odd number
        Z = (np.power(alp, 2.0) * np.power(r, 2.0) * (1.0 + np.exp(-alp * r * np.pi))) / ((np.power(k, 2.0) + np.power(alp, 2.0) * np.power(r, 2.0)) * (1.0 - np.exp(-alp * r * np.pi)))
    
    S1 = w / (1.0 - sig)
    S2 = (w + (ph / lam)) * Z - w
    S3 = sig - m * Z
    S = S1 + (S2 / S3)
    Gamk = -v * m * np.power(G, -m) * Z * S
    return Gamk

tau_space = np.linspace(0.01, 15.0, 255)
sigma_space = np.linspace(1.71, 7.0, 500)
ta, sig = np.meshgrid(tau_space, sigma_space)

frqs = [1,2,3,4,5,6] # frequency numbers
for k in frqs:
    fig, ax = plt.subplots()
    Gamk = eigenv(ta, sig, k)
    norm = mcolors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=Gamk.max())
    levels = np.linspace(Gamk.min(), Gamk.max(), 256)
    cont = ax.contour(ta, sig, Gamk, [0], linewidths=1.5)
    contf = ax.contourf(ta, sig, Gamk, levels=levels, cmap='jet', norm=norm)
    #cmap='rainbow' 'bwr' 'coolwarm' 'seismic'    
    ticks = np.linspace(Gamk.min(), Gamk.max(), 9)
    fig.colorbar(contf, extend='max').set_ticks(ticks)
    
    ax.set_aspect('auto', adjustable='box')

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\sigma$')
    plt.title('k={}'.format(k), fontsize=20)
    plt.savefig('sigma_heatmap_k_{}.png'.format(k), format='png', dpi=300)
    plt.show()
    pass
