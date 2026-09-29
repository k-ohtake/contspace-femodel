# contspace-femodel

This repository provides simulation code for the following paper.

I would appreciate your citing the following paper when you publish your results using this code.

---
Ohtake, K. (2025). A footloose entrepreneur model in a continuous space. arXiv preprint arXiv:2505.11241.  
<a href="https://doi.org/10.48550/arXiv.2505.11241" target="_blank" rel="noopener noreferrer">https://doi.org/10.48550/arXiv.2505.11241</a>

---

## Description

The "femodel" directory contains the following programs for analyzing the FE model.
(Similarly, the directory "cpmodel" and the directory "qllumodel" contain programs for the CP model and the QLLU model, respectively.) 
 
### fe_dynamics/fe_autoparam.ipynb  
This is code for simulating the evolution equation of the FE model.

Language:  
Julia ver 1.10.2  
Packages:  
CSV ver. 0.10.13  
DataFrames ver. 1.6.1  
Distributions ver. 0.25.107  
Format ver. 1.3.7  
IJulia ver. 1.24.2  
Plots ver. 1.40.2  

### fe_eigenvalue_homogeneous/fe_eigenvalue.py  
This is code for computing and plotting eigenvalues for the homogeneous stationary solution.

Language: Python ver 3.12.4  
Packages:  
matplotlib ver 3.10.1  
numpy 2.2.3  


### fe_eigenvalue_homogeneous/fe_heatmap.py  
This is code for computing eigenvalues and drawing heat maps of eigenvalues for the homogeneous stationary solution.

Language: Python ver 3.12.4  
Packages:  
matplotlib ver 3.10.1  
numpy 2.2.3  

### fe_eigenvalue_homogeneous/fe_contour.py
This is code for computing and plotting critical curves of eigenvalues for the homogeneous stationary solution.

Language: Python ver 3.12.4  
Packages:  
matplotlib ver 3.10.1  
numpy 2.2.3  

### fe_eigenvalue_spiky/fe_two-spikes-heatmap.py  
This is code for computing eigenvalues and drawing heat maps of eigenvalues for the stationary solution with two spikes.

Language: Python ver 3.12.4  
Packages:  
matplotlib ver 3.10.1  
numpy 2.2.3

### fe_eigenvalue_spiky/fe_three-spikes-heatmap.py  
This is code for computing eigenvalues and drawing heat maps of eigenvalues for the stationary solution with three spikes.

Language: Python ver 3.12.4  
Packages:  
matplotlib ver 3.10.7  
numpy 2.3.5
