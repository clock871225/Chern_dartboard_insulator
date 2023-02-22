from re import I
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

# make two dimensional tight-binding model
my_model=tb_model(2,2,lat,orb,nspin=1)

# set model parameters
pi=np.pi
i=1.j
m=1.0
t1=0.5
t2=0.5*i
t3=0.5

# set on-site energies
my_model.set_onsite([m,-m],mode='add')

# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t1, 0, 0, [ 2, 0])
my_model.set_hop(-t1, 1, 1, [ 2, 0])
my_model.set_hop(t1, 0, 0, [ 0, 2])
my_model.set_hop(-t1, 1, 1, [ 0, 2])
my_model.set_hop(t2, 0, 1, [ 2, 1])
my_model.set_hop(t2, 0, 1, [ -2,-1])
my_model.set_hop(-t2, 0, 1, [ 2, -1])
my_model.set_hop(-t2, 0, 1, [ -2, 1])
my_model.set_hop(t3, 0, 1, [ 1, 2])
my_model.set_hop(t3, 0, 1, [ -1, -2])
my_model.set_hop(-t3, 0, 1, [ 1, -2])
my_model.set_hop(-t3, 0, 1, [ -1, 2])


# generate object of type wf_array
N_en = 64 # number of spacing
my_array=wf_array(my_model, [N_en+1,N_en+1])
# solve model on a regular grid, and put origin of
# Brillouin zone at (0,0) point
my_array.solve_on_grid([0.0,0.0])

# compute Berry phases (Wilson-loop spectrum)
# if dir = 0(1), compute Berry phases along x(y)-direction
# n: order of CDI
cal_Berry_phase(N_en, my_array, dir = 0, n = 2)

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
(k_dist_2,k_node_2,evals_2) = cal_nano(my_model, dir = 0, N_nano = 50)
np.savez("CDI_2_edge.npz",k_dist_2,k_node_2,evals_2)

# ax.set_xlim(-1.5,1.5)
# ax.set_ylim(-1.5,1.5)
# fig.set_size_inches(5, 5)
# fig.tight_layout()

# x1 = np.linspace(-pi, pi, 21) 
# y1 = np.linspace(-pi, pi, 21) 
# X, Y = np.meshgrid(x1, y1)

# dz = 1.0+np.cos(2*X)+np.cos(2*Y)
# dx = -np.sin(X)*np.sin(2*Y)
# dy = np.sin(2*X)*np.sin(Y)
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
# fig.savefig('bqi.png',dpi=300)

plt.show()
