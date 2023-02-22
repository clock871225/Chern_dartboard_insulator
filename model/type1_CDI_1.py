from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt
from function.constant import *
from function.entanglement_spectrum import cal_es
from function.Berry_phase import cal_Berry_phase
from function.energy_spectrum import cal_energy
from function.nanoribbon import cal_nano
from function.corner import cal_corner

# define lattice vectors
lat=[[1.0,0.0],[0.0,1.0]]
# define coordinates of orbitals
orb=[[0.0,0.0],[0.0,0.0]]

# make two dimensional spinless tight-binding model
my_model=tb_model(2,2,lat,orb,nspin=1)

# set model parameters
m=0.0
t1=0.5
t2=-0.5*i
mu=0.0 # parameter to achieve quadratic band touching

# set on-site energies
my_model.set_onsite([0.5*mu**2,-0.5*mu**2])
my_model.set_onsite([m,-m],mode='add')

# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t1, 0, 0, [ 0, 1])
my_model.set_hop(-t1, 1, 1, [ 0, 1])
my_model.set_hop(t2, 0, 1, [ -1, 1])
my_model.set_hop(-t2, 0, 1, [ -1, -1])
my_model.set_hop(mu**2/4., 0, 0, [ 0, 1],mode='add')
my_model.set_hop(-mu**2/4., 1, 1, [ 0, 1],mode='add')
my_model.set_hop(0.5*mu, 0, 0, [ 1, 0])
my_model.set_hop(-0.5*mu, 1, 1, [ 1, 0])
my_model.set_hop(mu/4., 0, 0, [ 1, 1])
my_model.set_hop(mu/4., 0, 0, [ 1, -1])
my_model.set_hop(-mu/4., 1, 1, [ 1, 1])
my_model.set_hop(-mu/4., 1, 1, [ 1, -1])
my_model.set_hop(0.5*i*mu, 0, 1, [ 0, -1])
my_model.set_hop(-0.5*i*mu, 0, 1, [ 0, 1])

# parameters for Fig. 4

# my_model.set_onsite([0.2,-0.2],mode='add')
# my_model.set_hop(0.3, 0, 0, [ 1, 0],mode='add')
# my_model.set_hop(-0.3, 1, 1, [ 1, 0],mode='add')

# generate object of type wf_array
N_en = 64 # number of spacing
my_array=wf_array(my_model, [N_en+1,N_en+1])
# solve model on a regular grid, and put origin of
# Brillouin zone at (0,0) point
my_array.solve_on_grid([0.0,0.0])

# compute Berry phases (Wilson-loop spectrum)
# if dir = 0(1), compute Berry phases along x(y)-direction 
cal_Berry_phase(N_en, my_array, dir = 0)

# compute entanglement spectrum and energy 
# if dir = 0(1), compute the cut along x(y)-direction 
cal_es(N_en, my_array, dir = 0)

# compute energy spectrum
cal_energy(my_model)

# compute nanoribbon band structure
# if dir = 0(1), compute the nanoribbon along x(y)-direction
# N_nano is the width of the nanoribbon (number of unit cells)
# return: (list of k[], nodes of k[], band structures[band indices][k])
(k_dist_2,k_node_2,evals_2) = cal_nano(my_model, dir = 1, N_nano = 50)

# save files for plotting Fig. 3
# (k_dist_2,k_node_2,evals_2) = cal_nano(my_model, dir = 1, N_nano = 50)
# np.savez("CDI_1_y_edge.npz",k_dist_2,k_node_2,evals_2)

# N_nano is the width of the nanoflake (number of unit cells in x and y direction)
# return: (list of modes[], energies[], probability distribution of the corner mode[], flake model)
(n_list, evals_1, prob, flake_model) = cal_corner(my_model, N_corner = 10)

# plot Fig. 4 
# N_corner = 15

# (fig,ax)=flake_model.visualize(0,1,eig_dr=0.11/np.amax(prob)*prob,draw_hoppings=False,ph_color='black')
# ax.set_anchor((1.1,0.0))
# ax.axis('off')
# ax2=fig.add_subplot(121)
# ax2.set_position([-0.22,0.0, 0.5, 0.5])
# ax.annotate('b', xy=(0.0, 1.1), xycoords='axes fraction', fontsize=20,fontname='arial',fontfamily='sans-serif',fontweight=600)
# ax2.annotate('a', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20,fontname='arial',fontfamily='sans-serif',fontweight=600)
# fig.set_figheight(6)
# fig.set_figwidth(10)
# ax2.plot(n_list,evals_1,'bo',markersize=1)
# ax2.plot(449,0,'ro',markersize=5)
# ax2.plot(450,0,'ro',markersize=5)
# ax2.set_xlabel("state",fontsize=25)
# ax2.set_ylabel("E",fontsize=25)
# ax2.set_xlim(340,560)
# ax2.set_ylim(-1,1)
# ax2.xaxis.set_ticks([350,400,450,500,550])
# ax2.set_xticklabels([350,400,450,500,550],fontsize=18)
# ax2.yaxis.set_ticks([-1.0,-0.5,0.0,0.5,1.0])
# ax2.set_yticklabels([-1.0,-0.5,0.0,0.5,1.0],fontsize=18)
# fig.tight_layout()
# fig.savefig('corner.pdf')

# x1 = np.linspace(-pi, pi, 21) 
# y1 = np.linspace(-pi, pi, 21)  
# X, Y = np.meshgrid(x1, y1)
# dz = np.cos(Y)
# dx = np.sin(Y)*np.cos(X)
# dy = np.sin(Y)*np.sin(X)
# nor = (dx**2+dy**2+dz**2)**0.5
# dz = dz/nor
# dx = dx/nor
# dy = dy/nor

# fig, ax = plt.subplots() 
# im = ax.imshow(dz, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(-pi, pi, -pi, pi))
# ax.xaxis.set_ticks([-pi,0.0,pi])
# ax.set_xticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=20)
# ax.yaxis.set_ticks([-pi,0.0,pi])
# ax.set_yticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=20)
# cbar=fig.colorbar(im,ticks=[-1.0,-0.5,0.0,0.5,1.0],shrink=0.7,aspect=8)
# cbar.ax.tick_params(labelsize=15)
# for ia in range (21):
#   for ja in range (21):
#     ax.arrow(X[ja][ia],Y[ja][ia],dx[ja][ia]/4.,dy[ja][ia]/4.,width=0.02,head_width=0.1,color='black')

# fig.tight_layout()

plt.show()
