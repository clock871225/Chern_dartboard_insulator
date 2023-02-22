import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from function.constant import *

# plot energy spectrum
def cal_energy(my_model):

    print("calculating energy spectrum...")

    # generate list of k-points following a segmented path in the BZ
    # list of nodes (high-symmetry points) that will be connected
    path=[[0.,0.],[0.5,0.0],[0.5,0.5],[0.0,0.5], [0.,0.],[0.5,0.5]]

    # labels of the nodes
    label=(r'$\Gamma $',r'$X$', r'$M$', r'$Y$', r'$\Gamma $',r'$M$')
    (k_vec_3,k_dist_3,k_node_3) = my_model.k_path(path,251,report=False)

    # solve models
    (evals_3,evecs_3) = my_model.solve_all(k_vec_3,eig_vectors=True)

    # plot partial charge density for two orbitals
    points1 = np.array([k_dist_3, evals_3[0]]).T.reshape(-1, 1, 2)
    points2 = np.array([k_dist_3, evals_3[1]]).T.reshape(-1, 1, 2)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)

    fig, ax = plt.subplots()
    norm = plt.Normalize(0.0, 1.0)
    lc1 = LineCollection(segments1, cmap='viridis', norm=norm)
    lc2 = LineCollection(segments2, cmap='viridis', norm=norm)
    lc1.set_array(np.absolute(evecs_3[0,:,0]))
    lc2.set_array(np.absolute(evecs_3[1,:,0]))
    lc1.set_linewidth(3)
    lc2.set_linewidth(3)
    line1 = ax.add_collection(lc1)
    line2 = ax.add_collection(lc2)
    cbar = fig.colorbar(line1,ticks=[0.0,0.2,0.4,0.6,0.8,1.0],shrink=0.7,aspect=8)
    cbar.ax.tick_params(labelsize=15)
    ax.set_xlim(k_node_3[0],k_node_3[-1])
    ax.set_xticks(k_node_3)
    ax.set_xticklabels(label,fontsize=20)
    for n in range(len(k_node_3)):
        ax.axvline(x=k_node_3[n],linewidth=0.5, color='k')
    ax.set_ylabel("E",fontsize=25)
    ax.set_ylim(evals_3[0].min()-0.5,evals_3[1].max()+0.5)
    # ax.yaxis.set_ticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30])
    # ax.set_yticklabels((0.00,0.05,0.10,0.15,0.20,0.25,0.30),fontsize=18)
    fig.tight_layout()