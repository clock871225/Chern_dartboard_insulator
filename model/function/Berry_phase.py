import numpy as np
import matplotlib.pyplot as plt
from function.constant import *

# compute Berry phases along k_x for lower band
# if dir = 0(1), compute Berry phases along x(y)-direction
# n: order of CDI
def cal_Berry_phase(N_en,my_array, dir, n):

    print("calculating Berry phases...")

    phi_a_1 = my_array.berry_phase([0],dir,contin=True,berry_evals=True)

    # compute reduced Chern number

    print("calculating reduced Chern number...")

    flux_a_1 = -my_array.berry_flux([0],individual_phases=True)
    Cn =  0.0
    if n == 1:
        for ci in range (N_en):
            for cj in range (N_en//2):
                Cn += flux_a_1[ci][cj]
    elif n == 2:
        for ci in range (N_en//2):
            for cj in range (N_en//2):
                Cn += flux_a_1[ci][cj]
    print ("reduced Chern number =",Cn/2/pi)

    # plot Berry phases

    fig, ax = plt.subplots()
    k=np.linspace(0.,1.,len(phi_a_1))
    ax.plot(k,phi_a_1[:], 'r')
    ax.plot(k,phi_a_1[:]+2*pi, 'r')
    ax.plot(k,phi_a_1[:]-2*pi, 'r')
    ax.axhline(y=-pi,linewidth=1.0, color='k')
    ax.axhline(y=pi,linewidth=1.0, color='k')
    #ax.set_title("Wilson-loop Spectrum",fontsize=25)
    ax.set_xlabel(r"$k$",fontsize=25)
    ax.set_ylabel(r"$\theta$",fontsize=25)
    ax.set_xlim(0.,1.)
    ax.xaxis.set_ticks([0.,0.5,1.])
    ax.set_xticklabels((r'$0$',r'$\pi$', r'$2\pi$'),fontsize=18)
    ax.set_ylim(-2*pi,2*pi)
    ax.yaxis.set_ticks([-2.*np.pi,-np.pi,0.,np.pi,2.*np.pi])
    ax.set_yticklabels((r'$-2\pi$',r'$-\pi$',r'$0$',r'$\pi$', r'$2\pi$'),fontsize=18)
    fig.tight_layout()