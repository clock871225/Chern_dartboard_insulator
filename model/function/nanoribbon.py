import numpy as np
import matplotlib.pyplot as plt
from function.constant import *

# compute nanoribbon band structure
# if dir = 0(1), compute the nanoribbon along x(y)-direction
# N_nano is the width of the nanoribbon (number of unit cells)
# return: (list of k[], nodes of k[], band structures[band indices][k])
# n: order of CDI
def cal_nano(my_model, dir, N_nano, n):

    print("calculating nanoribbon band structure...")

    # cutout ribbon model
    if dir == 0 and n != 3:
        temp_model = my_model.make_supercell([[1,0],[0,N_nano]],to_home=True)
    elif dir == 0 and n != 3:
        temp_model = my_model.make_supercell([[0,1],[-N_nano,0]],to_home=True)
    elif dir == 0 and n == 3:
        temp_model = my_model.make_supercell([[1,0],[0,N_nano]],to_home=True)
    elif dir == 1 and n == 3:
        temp_model = my_model.make_supercell([[1,1],[0,N_nano]],to_home=True)
    
    ribbon_model = temp_model.cut_piece(1,1,glue_edgs=False)

    # generate k-list and solve models
    (k_vec_2,k_dist_2,k_node_2) = ribbon_model.k_path('full',201,report=False)
    evals_2 = ribbon_model.solve_all(k_vec_2)

    # plot energy spectrum of nanonribbon
    fig, ax = plt.subplots()

    for n in range(evals_2.shape[0]):
        ax.plot(k_dist_2,evals_2[n],"k-")

    # highlight the edge states
    # ax.plot(k_dist_2,evals_2[N_nano+1],"r-")
    # ax.plot(k_dist_2,evals_2[N_nano],"r-")

    ax.set_xlim(k_dist_2[0],k_dist_2[-1])
    #ax.set_ylim(-1.0,1.0)
    #ax.set_title("Nanoribbon Band Structure",fontsize=25)
    ax.set_xlabel("k",fontsize=25)
    ax.set_ylabel("E",fontsize=25)
    #ax.yaxis.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
    #ax.set_yticklabels((-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0),fontsize=18)
    ax.xaxis.set_ticks(k_node_2)
    ax.set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)
    fig.tight_layout()

    return (k_dist_2,k_node_2,evals_2)