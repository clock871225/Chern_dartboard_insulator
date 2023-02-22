import numpy as np
import matplotlib.pyplot as plt
from constant import *

# compute nanoflake energy spectrum and plot corner modes
def cal_corner(my_model, N_corner):

    print("calculating nanoflake energy spectrum...")

    # make the supercell of the model
    sc_model = my_model.make_supercell([[N_corner, -N_corner], [N_corner, N_corner]], to_home=True)

    # cutout the supercell
    slab_model = sc_model.cut_piece(1, 1, glue_edgs=False)
    flake_model = slab_model.cut_piece(1, 0, glue_edgs=False)

    # solve models
    (evals_1,evecs)=flake_model.solve_all(eig_vectors=True)

    # pick index of state in the middle of the gap

    total = flake_model.get_num_orbitals()
    ed = total//2
    ed_n = ed-1
    print("corner mode energy =", evals_1[ed_n])

    # compute the probability distribution of the wavefuncion

    prob = np.zeros(ed)
    prob = np.multiply(np.conjugate(evecs[ed_n,:]),evecs[ed_n,:])
    n_list = np.arange(total)

    # plot energy spectrum of nanoflake

    fig, ax = plt.subplots()
    ax.plot(n_list,evals_1,'bo',markersize=1)

    # highlight the corner states
    ax.plot(ed_n,0,'ro',markersize=5)
    ax.plot(ed,0,'ro',markersize=5)

    #ax.set_title("Nanoflake Energy Spectrum",fontsize=25)
    ax.set_xlabel("state",fontsize=25)
    ax.set_ylabel("E",fontsize=25)
    # ax.set_xlim(340,560)
    # ax.set_ylim(-1,1)
    # ax.xaxis.set_ticks([350,400,450,500,550])
    # ax.set_xticklabels([350,400,450,500,550],fontsize=18)
    # ax.yaxis.set_ticks([-1.0,-0.5,0.0,0.5,1.0])
    # ax.set_yticklabels([-1.0,-0.5,0.0,0.5,1.0],fontsize=18)
    fig.tight_layout()

    # draw the edge state
    (fig,ax)=flake_model.visualize(0,1,eig_dr=0.2/np.amax(prob)*prob,draw_hoppings=False,ph_color='black')
    #ax.set_title("Midgap Corner State",fontsize=25)
    #ax.set_xlabel("x coordinate")
    #ax.set_ylabel("y coordinate")
    fig.tight_layout()

    return (n_list, evals_1, prob, flake_model)