from pythtb import * # import TB model class
import numpy as np
from scipy.linalg import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from entanglement_spectrum import cal_es

# define lattice vectors
lat=[[1.0,0.0],[0.0,1.0]]
# define coordinates of orbitals
orb=[[0.0,0.0],[0.0,0.0]]

# make two dimensional tight-binding model
my_model=tb_model(2,2,lat,orb,nspin=1)

# set model parameters
pi=np.pi
i=1.j
m=0.0
mu=0.0
t1=0.5
t2=-0.5*i

# set on-site energies
my_model.set_onsite([0.5*m**2,-0.5*m**2])
my_model.set_onsite([mu,-mu],mode='add')

# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t1, 0, 0, [ 0, 1])
my_model.set_hop(-t1, 1, 1, [ 0, 1])
my_model.set_hop(t2, 0, 1, [ -1, 1])
my_model.set_hop(-t2, 0, 1, [ -1, -1])
my_model.set_hop(m**2/4., 0, 0, [ 0, 1],mode='add')
my_model.set_hop(-m**2/4., 1, 1, [ 0, 1],mode='add')
my_model.set_hop(0.5*m, 0, 0, [ 1, 0])
my_model.set_hop(-0.5*m, 1, 1, [ 1, 0])
my_model.set_hop(m/4., 0, 0, [ 1, 1])
my_model.set_hop(m/4., 0, 0, [ 1, -1])
my_model.set_hop(-m/4., 1, 1, [ 1, 1])
my_model.set_hop(-m/4., 1, 1, [ 1, -1])
my_model.set_hop(0.5*i*m, 0, 1, [ 0, -1])
my_model.set_hop(-0.5*i*m, 0, 1, [ 0, 1])

# my_model.set_hop(0.1+0.1*i, 0, 1, [ 0, 1],mode='add')
# my_model.set_hop(-0.1-0.1*i, 0, 1, [ 0, -1],mode='add')

# my_model.set_hop(0.2+0.2*i, 0, 1, [ -1, 1],mode='add')
# my_model.set_hop(-0.2-0.2*i, 0, 1, [ -1, -1],mode='add')

# my_model.set_hop(0.3, 0, 0, [ 1, 0],mode='add')
# my_model.set_hop(-0.3, 1, 1, [ 1, 0],mode='add')

# generate object of type wf_array
N_en = 64 # number of spacing
my_array=wf_array(my_model,[N_en+1,N_en+1])
# solve model on a regular grid, and put origin of
# Brillouin zone at (0,0) point
my_array.solve_on_grid([0.0,0.0])

# Berry phases along k_x for lower band
phi_a_1 = my_array.berry_phase([0],1,contin=True,berry_evals=True)
flux_a_1 = -my_array.berry_flux([0],individual_phases=True)
C =  0.0
for ci in range (N_en):
    for cj in range (N_en//2):
      C += flux_a_1[ci][cj]
print (C/pi)

# compute the entanglement spectrum and energy 
# if dir = 0(1), compute the cut along x(y)-direction 
cal_es(N_en,my_array,dir = 0) 


# cutout ribbon model
temp_model=my_model.make_supercell([[0,1],[-50,0]],to_home=True)
ribbon_model=temp_model.cut_piece(1,1,glue_edgs=False)

# make the supercell of the model
sc_model=my_model.make_supercell([[10,-10],[10,10]],to_home=True)

# cutout the supercell
slab_model=sc_model.cut_piece(1,1,glue_edgs=False)
flake_model=slab_model.cut_piece(1,0,glue_edgs=False)

# solve models
(evals_1,evecs)=flake_model.solve_all(eig_vectors=True)
(k_vec_2,k_dist_2,k_node_2)=ribbon_model.k_path('full',201,report=False)
evals_2=ribbon_model.solve_all(k_vec_2)

# generate list of k-points following a segmented path in the BZ
# list of nodes (high-symmetry points) that will be connected
path=[[0.,0.],[0.5,0.0],[0.5,0.5],[0.0,0.5], [0.,0.],[0.5,0.5]]
# labels of the nodes
label=(r'$\Gamma $',r'$X$', r'$M$', r'$Y$', r'$\Gamma $',r'$M$')
(k_vec_3,k_dist_3,k_node_3)=my_model.k_path(path,251,report=False)
(evals_3,evecs_3)=my_model.solve_all(k_vec_3,eig_vectors=True)

# pick index of state in the middle of the gap
total=flake_model.get_num_orbitals()
ed=total//2
ed_n = ed-1
print(evals_1[ed_n])

# compute the probability distribution of the wavefuncion
prob = np.zeros(ed)

prob=np.multiply(np.conjugate(evecs[ed_n,:]),evecs[ed_n,:])

x=np.arange(total)
ex_3=np.arange(N_en+1)/N_en*2*pi
 
# draw the edge state
(fig,ax)=flake_model.visualize(0,1,eig_dr=0.2/np.amax(prob)*prob,draw_hoppings=False,ph_color='black')
#ax.set_title("Midgap Corner State",fontsize=25)
#ax.set_xlabel("x coordinate")
#ax.set_ylabel("y coordinate")
fig.tight_layout()

# plot energy spectrum of nanoflake
fig, ax = plt.subplots()
ax.plot(x,evals_1,'bo',markersize=1)
# ax.plot(449,0,'ro',markersize=5)
# ax.plot(450,0,'ro',markersize=5)
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

# (fig,ax)=flake_model.visualize(0,1,eig_dr=0.11/np.amax(prob)*prob,draw_hoppings=False,ph_color='black')
# ax.set_anchor((1.1,0.0))
# ax.axis('off')
# ax2=fig.add_subplot(121)
# ax2.set_position([-0.22,0.0, 0.5, 0.5])
# ax.annotate('b', xy=(0.0, 1.1), xycoords='axes fraction', fontsize=20,fontname='arial',fontfamily='sans-serif',fontweight=600)
# ax2.annotate('a', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20,fontname='arial',fontfamily='sans-serif',fontweight=600)
# fig.set_figheight(6)
# fig.set_figwidth(10)
# ax2.plot(x,evals_1,'bo',markersize=1)
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

# plot energy spectrum of nanonribbon
fig, ax = plt.subplots()

# ax.plot(k_dist_2,evals_2[51],"r-")
# ax.plot(k_dist_2,evals_2[50],"r-")
for n in range(evals_2.shape[0]):
  ax.plot(k_dist_2,evals_2[n],"k-")

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

# plot energy spectrum
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
cbar=fig.colorbar(line1,ticks=[0.0,0.2,0.4,0.6,0.8,1.0],shrink=0.7,aspect=8)
cbar.ax.tick_params(labelsize=15)

ax.set_xlim(k_node_3[0],k_node_3[-1])
ax.set_xticks(k_node_3)
ax.set_xticklabels(label,fontsize=20)
for n in range(len(k_node_3)):
  ax.axvline(x=k_node_3[n],linewidth=0.5, color='k')
#ax.set_title("det(s)",fontsize=25)
ax.set_ylabel("E",fontsize=25)
ax.set_ylim(evals_3[0].min()-0.5,evals_3[1].max()+0.5)
# ax.yaxis.set_ticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30])
# ax.set_yticklabels((0.00,0.05,0.10,0.15,0.20,0.25,0.30),fontsize=18)
fig.tight_layout()

# plot Wannier function
# fig, ax = plt.subplots()
# ax.set_title("Wannier Function",fontsize=25)
# ax.set_xlabel(r"r/a",fontsize=25)
# ax.set_ylabel(r"$log_{10}(|W(r)|^2)$",fontsize=25)
# ax.plot(r,pw,'bo',markersize=3)
# ax.set_xlim(0,31)
# ax.set_ylim(-31,0)
# ax.xaxis.set_ticks([0,5,10,15,20,25,30])
# ax.set_xticklabels((0,5,10,15,20,25,30),fontsize=18)
# ax.yaxis.set_ticks([0.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0])
# ax.set_yticklabels((0.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0),fontsize=18)
# fig.tight_layout()
# fig.savefig("fig19.png")

# msigmax=np.array([[0,1],[1,0]],dtype=complex)
# msigmay=np.array([[0,-i],[i,0]],dtype=complex)

#plot Wannier function
# fig, ax = plt.subplots()
# ax.set_title("Wannier Function",fontsize=25)
# ax.xaxis.set_ticks([-1.,0.,1.])
# ax.set_xticklabels((r'-a',r'0', r'a'),fontsize=20)
# ax.yaxis.set_ticks([-1.,0.,1.])
# ax.set_yticklabels((r'-a',r'0', r'a'),fontsize=20)
# for nwx in range (-1,2):
#   for nwy in range (-1,2):
#     sw0 = np.array([sw[Np*nwx+nwy+4][0],sw[Np*nwx+nwy+4][1]],dtype=complex)
#     sx = np.real(np.vdot(sw0,np.matmul(msigmax,sw0)))*10
#     sy = np.real(np.vdot(sw0,np.matmul(msigmay,sw0)))*10
#     if nwx!=0 or nwy!=0:
#       ax.arrow(nwx,nwy,sx,sy,width=0.03,head_width=0.1,color='blue')
#     pw = np.real(np.vdot(sw0,sw0))
#     ax.plot(nwx,nwy,'ro',markersize=50*(pw**0.5))

# ax.set_xlim(-1.5,1.5)
# ax.set_ylim(-1.5,1.5)
# fig.set_size_inches(5, 5)
# fig.tight_layout()

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
