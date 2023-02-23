import numpy as np
from scipy.linalg import *
import matplotlib.pyplot as plt
from function.constant import *

# compute entanglement spectrum and energy 
# if dir = 0(1), compute the cut along x(y)-direction 
def cal_es(N_en,my_array, dir):

    print("calculating entanglement spectrum...")
    
    h_list = np.arange(N_en//2,N_en) # define the half list from N_en//2 to N_en
    correlation = np.zeros((N_en+1,N_en,N_en),dtype=complex) # initialize the correlation matrix for each k
    exp_matrix = np.zeros((N_en,N_en//2,N_en//2),dtype=complex) # initalize the exponential matrix for each k 
    es_set = np.zeros((N_en+1,N_en)) # initialize the entanglement spectrum for each k

    for k1 in range (N_en):
        ex_h_list = np.exp(-i*2.0*pi/N_en*k1*h_list) # exponentialize the half list
        exp_matrix[k1]  = np.outer(np.conjugate(ex_h_list), ex_h_list)/N_en # compute the exponential matrix for each k

    for i1 in range (N_en+1):
        for i2 in range (N_en):
            if dir == 0:
                wf_vec = np.ndarray.flatten(my_array[i1,i2][0]) # define the wavefunction in terms of vector for each kx and ky
            else:
                wf_vec = np.ndarray.flatten(my_array[i2,i1][0]) # define the wavefunction in terms of vector for each ky and kx
            
            wf_projector  = np.outer(wf_vec,np.conjugate(wf_vec)) # compute the projector of wavefunction for each kx and ky
            correlation_k = np.kron(exp_matrix[i2], wf_projector) # compute the correlation matrix for each kx and ky
            correlation[i1] += correlation_k # compute the correlation matrix for each kx (ky)

        es_set[i1] = np.sort(np.real(eigvals(correlation[i1]))) # compute the eigenvalues (entanglement spectrum) and sort

    energy_set = 0.5*np.log(1.0/es_set-1.0) # compute the entanglement energy

    # plot the entanglement spectrum 

    list_k = np.arange(N_en+1)/N_en*2*pi
    fig, ax = plt.subplots()
    for n in range(N_en):
        ax.plot(list_k,es_set[:,n],"k-")
    ax.set_xlim(0,2*pi)
    #ax.set_ylim(-1.0,1.0)
    #ax.set_title("Entanglement Spectrum",fontsize=25)
    ax.set_xlabel("k",fontsize=25)
    ax.set_ylabel(r'$\xi$',fontsize=25)
    ax.xaxis.set_ticks([0,pi,2*pi])
    ax.set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)
    #ax.yaxis.set_ticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    #ax.set_yticklabels((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),fontsize=18)
    fig.tight_layout()

    # plot the entanglement energy

    fig, ax = plt.subplots()
    for n in range(N_en):
        ax.plot(list_k,energy_set[:,n],"k-")
    ax.set_xlim(0,2*pi)
    ax.set_ylim(-15,15)
    #ax.set_title("Entanglement Energy",fontsize=25)
    ax.set_xlabel("k",fontsize=25)
    ax.set_ylabel(r'$\epsilon$',fontsize=25)
    ax.xaxis.set_ticks([0,pi,2*pi])
    ax.set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)
    #ax.yaxis.set_ticks([-15,-10,-5,0,5,10,15])
    #ax.set_yticklabels((-15,-10,-5,0,5,10,15),fontsize=18)
    fig.tight_layout()




