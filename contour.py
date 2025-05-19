# -*- coding: utf-8 -*-
"""
2025.4.10 thu. created by Kensuke Ohtake
Footloose-Entrepreneur model
Eigenvalues analysis
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

tau_space = np.linspace(0.01, 15.0, 1024)#0.8
sigma_space = np.linspace(1.61, 7.0, 1024)#7.0
ta, sig = np.meshgrid(tau_space, sigma_space)

frqs = [1,2,3,4,5,6]
labels = []
hs = []
fig, ax = plt.subplots()
i = 0
for k in frqs:
    Gamk = eigenv(ta, sig, k)
    cont = ax.contour(ta, sig, Gamk, [0], linewidths=1.5, colors=[matplotlib.cm.tab10(i)])
    lb = r'$k$ = {}'.format(k)
    labels.append(lb)
    h,_ = cont.legend_elements()
    hs.append(h[0])
    i += 1
    pass

ax.legend(hs, labels)
plt.gca().set_aspect('equal', adjustable='box')
ax.set_facecolor('0.95')
ax.set_aspect('auto', adjustable='box')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma$')
plt.grid(linestyle=':')
plt.savefig('contours.png', format='png', dpi=300)
plt.show()
