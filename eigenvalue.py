# -*- coding: utf-8 -*-
"""
2025.4.10 thu. created by Kensuke Ohtake
Footloose-Entrepreneur model
Eigenvalues analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

# get datetime
DateTime = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

# set parameters
m = 0.6 # mu
Lam = 1.0 # total mobile workers
Ph = 1.0 # total immobile workers
r = 1.0 # radius
F = 1.0 # fixed input
v = 1.0 # adjustment speed
lam = Lam/(2.0*np.pi*r) # homogeneous mobile population
ph = Ph/(2.0*np.pi*r) # homogeneous immobile population

def eigenv(X, Y, k):

    # X: tau vector
    # Y: sigma
    # k: frequency number (k)
    
    k = float(k)

    alp = (Y - 1.0) * X # alpha vector
    w = ((m * ph)/(sig * lam)) / (1.0 - (m / sig)) # homogeneous nomilal wage
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

ta = np.linspace(0.01, 3.0, 300) # tau
sig = 3.0 # sigma

frqs = [1,2,3,4,5,6] # frequency numbers
fig, ax = plt.subplots()
for k in frqs:
    Gamk = eigenv(ta, sig, k)
    plt.plot(ta, Gamk, label='k={}'.format(k))
    pass

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Gamma_k$')
plt.ylim(-0.01, 0.015)
plt.axhline(y=0, color='gray', linestyle='dotted')
#ax.yaxis.set_ticks_position('left')
#ax.spines['left'].set_position(('data', 0))
ax.set_facecolor('0.95')
plt.grid(linestyle=':')
plt.legend()
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.savefig('eigenvalues.png', format='png', dpi=300)
plt.show()
