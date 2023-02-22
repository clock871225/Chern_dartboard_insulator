from re import I
from pythtb import * # import TB model class
import numpy as np
from scipy.linalg import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# define lattice vectors
lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
# define coordinates of orbitals
orb=[[0.0,0.0],[0.0,0.0]]

# make two dimensional tight-binding model
my_model=tb_model(2,2,lat,orb,nspin=1)

# set model parameters
pi=np.pi
i=1.j
m=1.5
t1=1.0 
t3=0.0 
t5=1.0
t2=0.0 
t4=0.0
t6=0.5
t7=0.0


# set on-site energies
my_model.set_onsite([m,-m],mode='add')

# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t1, 0, 0, [ 1, 0])
my_model.set_hop(-t1, 1, 1, [ 1, 0])
my_model.set_hop(t1, 0, 0, [ 0, 1])
my_model.set_hop(-t1, 1, 1, [ 0, 1])
my_model.set_hop(t1, 0, 0, [ -1, 1])
my_model.set_hop(-t1, 1, 1, [ -1, 1])
my_model.set_hop(t3, 0, 0, [ 1, 1])
my_model.set_hop(-t3, 1, 1, [ 1, 1])
my_model.set_hop(t3, 0, 0, [ -1, 2])
my_model.set_hop(-t3, 1, 1, [ -1, 2])
my_model.set_hop(t3, 0, 0, [ -2, 1])
my_model.set_hop(-t3, 1, 1, [ -2, 1])
my_model.set_hop(t5, 0, 0, [ 2, 0])
my_model.set_hop(-t5, 1, 1, [ 2, 0])
my_model.set_hop(t5, 0, 0, [ 0, 2])
my_model.set_hop(-t5, 1, 1, [ 0, 2])
my_model.set_hop(t5, 0, 0, [ -2, 2])
my_model.set_hop(-t5, 1, 1, [ -2, 2])
# my_model.set_hop(t2, 0, 1, [ 1, 0])
# my_model.set_hop(-t2, 0, 1, [ -1, 0])
# my_model.set_hop(-t2, 0, 1, [ 0, 1])
# my_model.set_hop(t2, 0, 1, [ 0, -1])
# my_model.set_hop(t2, 0, 1, [ -1, 1])
# my_model.set_hop(-t2, 0, 1, [ 1, -1])
# my_model.set_hop(t4, 0, 1, [ 2, 0])
# my_model.set_hop(-t4, 0, 1, [ -2, 0])
# my_model.set_hop(-t4, 0, 1, [ 0, 2])
# my_model.set_hop(t4, 0, 1, [ 0, -2])
# my_model.set_hop(t4, 0, 1, [ -2, 2])
# my_model.set_hop(-t4, 0, 1, [ 2, -2])
my_model.set_hop(t6, 0, 1, [ 2, 1])
my_model.set_hop(-t6, 0, 1, [ 1, 2])
my_model.set_hop(-t6, 0, 1, [ -3, 1])
my_model.set_hop(t6, 0, 1, [ -3, 2])
my_model.set_hop(t6, 0, 1, [ 1, -3])
my_model.set_hop(-t6, 0, 1, [ 2, -3])
# my_model.set_hop(t7, 0, 1, [ -1, 3])
# my_model.set_hop(-t7, 0, 1, [ -2, 3])
# my_model.set_hop(-t7, 0, 1, [ 3, -1])
# my_model.set_hop(t7, 0, 1, [ 3, -2])
# my_model.set_hop(t7, 0, 1, [ -2, -1])
# my_model.set_hop(-t7, 0, 1, [ -1, -2])


# generate object of type wf_array that will be used for
# Berry phase and curvature calculations
N_en = 64
my_array=wf_array(my_model,[N_en+1,N_en+1])
# solve model on a regular grid, and put origin of
# Brillouin zone at -1/2 -1/2 point
my_array.solve_on_grid([0.0,0.0])

# Berry phases along k_x for lower band
phi_a_1 = my_array.berry_phase([0],0,contin=True,berry_evals=True)
flux_a_1 = -my_array.berry_flux([0],individual_phases=True)
C1 = 0.0
for cj in range (N_en):
    for ci in range (N_en-cj):
      C1 += flux_a_1[ci][cj+ci]

print (C1/pi)

a = np.array([1,0])/2.
b = np.array([0,1])/2.
wk = np.zeros((N_en+1,N_en,2),dtype=complex)

# compute entanglement spectrum

ex_1=np.arange(N_en//2,N_en)
p = np.zeros((N_en+1,N_en,N_en),dtype=complex)
mex = np.zeros((N_en,N_en//2,N_en//2),dtype=complex)
en_set = np.zeros((N_en+1,N_en))

for j1 in range (N_en):
  ex_2 = np.exp(-i*2.0*pi/N_en*j1*ex_1)
  mex[j1]  = np.outer(np.conjugate(ex_2), ex_2)/N_en

Np = 1
sw = np.zeros((Np*Np,2),dtype=complex)
r = np.zeros(Np*Np)
pw = np.zeros(Np*Np)
for nx in range (Np):
  for ny in range (Np):
    wx = np.zeros(2,dtype=complex)
    for i1 in range (N_en+1):
      for i2 in range (N_en): 
        mp0 = np.ndarray.flatten(my_array[i1,i2][0])
        mp  = np.outer(mp0,np.conjugate(mp0))
        if nx==0 and ny==0:
          p_k = np.kron(mex[i2], mp)
          p[i1] += p_k
        trial = b
        S  = np.vdot(trial,np.matmul(mp,trial))
        wk[i1][i2] = S**(-0.5)*np.matmul(mp,trial)
        if i1 != N_en:
          wx += wk[i1][i2]*np.exp(i*(nx*i1+ny*i2)*2*pi/N_en)/(N_en**2)
      if nx==0 and ny==0:    
        en_set[i1] = np.sort(np.real(eigvals(p[i1])))

    # sw[Np*nx+ny+4] = wx
    r[Np*nx+ny] = (nx**2+ny**2)**0.5
    pw[Np*nx+ny] = np.log10(np.real(np.vdot(wx,wx)))

energy_set = 0.5*np.log(1.0/en_set-1.0)

# cutout ribbon model
temp_model=my_model.make_supercell([[1,0],[0,50]],to_home=True)
ribbon_model=temp_model.cut_piece(1,1,glue_edgs=False)

# make the supercell of the model
sc_model=my_model.make_supercell([[10,0],[0,10]],to_home=True)

# cutout the supercell
slab_model=sc_model.cut_piece(1,1,glue_edgs=False)
flake_model=slab_model.cut_piece(1,0,glue_edgs=False)

# solve models
(evals_1,evecs)=flake_model.solve_all(eig_vectors=True)
(k_vec_2,k_dist_2,k_node_2)=ribbon_model.k_path('full',201,report=False)
(evals_2,evecs_2)=ribbon_model.solve_all(k_vec_2,eig_vectors=True)
# print(evecs_2[50][168])

# generate list of k-points following a segmented path in the BZ
# list of nodes (high-symmetry points) that will be connected
path=[[0.,0.],[0.,0.5],[1.0/3.,2.0/3.], [0.5,0.5],[0.,0.0]]
# labels of the nodes
label=(r'$\Gamma $',r'$M$', r'$K$', r'$M^\prime$', r'$\Gamma $')
(k_vec_3,k_dist_3,k_node_3)=my_model.k_path(path,201,report=False)
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
# ax.plot(1600,0,'ro',markersize=5)
# ax.plot(1599,0,'ro',markersize=5)
#ax.set_title("Nanoflake Energy Spectrum",fontsize=25)
ax.set_xlabel("state",fontsize=25)
ax.set_ylabel("E",fontsize=25)
# ax.set_xlim(1500,1700)
# ax.set_ylim(-1,1)
# ax.xaxis.set_ticks([1500,1550,1600,1650,1700])
# ax.set_xticklabels([1500,1550,1600,1650,1700],fontsize=18)
# ax.yaxis.set_ticks([-1.0,-0.5,0.0,0.5,1.0])
# ax.set_yticklabels([-1.0,-0.5,0.0,0.5,1.0],fontsize=18)
fig.tight_layout()


# n3_1=np.load('n=3-2 edge-2.npz')
# k_dist_2_3_1=n3_1['arr_0']
# k_node_2_3_1=n3_1['arr_1']
# evals_2_3_1=n3_1['arr_2']
# fig, ax = plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios': [1, 1]}) 

# for n in range(evals_2.shape[0]):
#   ax[0].plot(k_dist_2,evals_2[n],"0.5")
#   ax[1].plot(k_dist_2_3_1,evals_2_3_1[n],"0.5")

# ax[0].plot(k_dist_2[15:32],evals_2[49][15:32],"b-")
# ax[0].plot(k_dist_2[31:37],evals_2[50][31:37],"b-")
# ax[0].plot(k_dist_2[57:72],evals_2[51][57:72],"b-")
# ax[0].plot(k_dist_2[71:79],evals_2[50][71:79],"b-")
# ax[0].plot(k_dist_2[78:123],evals_2[49][78:123],"b-")
# ax[0].plot(k_dist_2[122:130],evals_2[50][122:130],"b-")
# ax[0].plot(k_dist_2[129:144],evals_2[51][129:144],"b-")
# ax[0].plot(k_dist_2[164:170],evals_2[50][164:170],"b-")
# ax[0].plot(k_dist_2[169:186],evals_2[49][169:186],"b-")
# ax[0].plot(k_dist_2[36:52],evals_2[49][36:52],"b-")
# ax[0].plot(k_dist_2[51:72],evals_2[50][51:72],"b-")
# ax[0].plot(k_dist_2[71:88],evals_2[51][71:88],"b-")
# ax[0].plot(k_dist_2[113:130],evals_2[51][113:130],"b-")
# ax[0].plot(k_dist_2[129:150],evals_2[50][129:150],"b-")
# ax[0].plot(k_dist_2[149:165],evals_2[49][149:165],"b-")

# ax[0].plot(k_dist_2[15:32],evals_2[50][15:32],"k-")
# ax[0].plot(k_dist_2[31:37],evals_2[49][31:37],"k-")
# ax[0].plot(k_dist_2[57:72],evals_2[48][57:72],"k-")
# ax[0].plot(k_dist_2[71:79],evals_2[49][71:79],"k-")
# ax[0].plot(k_dist_2[78:123],evals_2[50][78:123],"k-")
# ax[0].plot(k_dist_2[122:130],evals_2[49][122:130],"k-")
# ax[0].plot(k_dist_2[129:144],evals_2[48][129:144],"k-")
# ax[0].plot(k_dist_2[164:170],evals_2[49][164:170],"k-")
# ax[0].plot(k_dist_2[169:186],evals_2[50][169:186],"k-")
# ax[0].plot(k_dist_2[36:52],evals_2[50][36:52],"k-")
# ax[0].plot(k_dist_2[51:72],evals_2[49][51:72],"k-")
# ax[0].plot(k_dist_2[71:88],evals_2[48][71:88],"k-")
# ax[0].plot(k_dist_2[113:130],evals_2[48][113:130],"k-")
# ax[0].plot(k_dist_2[129:150],evals_2[49][129:150],"k-")
# ax[0].plot(k_dist_2[149:165],evals_2[50][149:165],"k-")

# ax[1].plot(k_dist_2_3_1,evals_2_3_1[48],"r-")
# ax[1].plot(k_dist_2_3_1,evals_2_3_1[49],"r-")
# ax[1].plot(k_dist_2_3_1,evals_2_3_1[50],"r-")
# ax[1].plot(k_dist_2_3_1,evals_2_3_1[51],"r-")


# ax[0].set_xlim(k_dist_2[0],k_dist_2[-1])
# ax[0].set_ylim(-3.5,3.5)
# ax[0].set_xlabel("k",fontsize=25)
# ax[0].set_ylabel("E",fontsize=25)
# ax[0].yaxis.set_ticks([-3,-2,-1,0,1,2,3])
# ax[0].set_yticklabels((-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0),fontsize=18)
# ax[0].xaxis.set_ticks(k_node_2)
# ax[0].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)

# ax[1].set_xlim(k_dist_2_3_1[0],k_dist_2_3_1[-1])
# ax[1].set_ylim(-3.5,3.5)
# ax[1].set_xlabel("wave vector",fontsize=25)
# ax[1].set_ylabel("energy",fontsize=25)
# ax[1].yaxis.set_ticks([-3,-2,-1,0,1,2,3])
# ax[1].set_yticklabels((-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0),fontsize=18)
# ax[1].xaxis.set_ticks(k_node_2_3_1)
# ax[1].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)

# ax[0].annotate('a', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20,fontname='arial',fontfamily='sans-serif',fontweight=600)
# ax[1].annotate('b', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=20,fontname='arial',fontfamily='sans-serif',fontweight=600)

# fig.tight_layout()

# fig.savefig('n=3-2edge.pdf')

# plot energy spectrum of nanonribbon
fig, ax = plt.subplots()

for n in range(evals_2.shape[0]):
  ax.plot(k_dist_2,evals_2[n],"k-")

ax.plot(k_dist_2,evals_2[48],"r-")
ax.plot(k_dist_2,evals_2[49],"r-")
ax.plot(k_dist_2,evals_2[50],"r-")
ax.plot(k_dist_2,evals_2[51],"r-")
# ax.plot(k_dist_2[0:32],evals_2[49][0:32],"b-")
# ax.plot(k_dist_2[31:37],evals_2[50][31:37],"b-")
# ax.plot(k_dist_2[36:72],evals_2[51][36:72],"b-")
# ax.plot(k_dist_2[71:79],evals_2[50][71:79],"b-")
# ax.plot(k_dist_2[78:101],evals_2[49][78:101],"b-")
# ax.plot(k_dist_2[100:123],evals_2[49][100:123],"b-")
# ax.plot(k_dist_2[122:130],evals_2[50][122:130],"b-")
# ax.plot(k_dist_2[129:165],evals_2[51][129:165],"b-")
# ax.plot(k_dist_2[164:170],evals_2[50][164:170],"b-")
# ax.plot(k_dist_2[169:200],evals_2[49][169:200],"b-")
# ax.plot(k_dist_2[0:37],evals_2[48][0:37],"b-")
# ax.plot(k_dist_2[36:52],evals_2[49][36:52],"b-")
# ax.plot(k_dist_2[51:72],evals_2[50][51:72],"b-")
# ax.plot(k_dist_2[71:130],evals_2[51][71:130],"b-")
# ax.plot(k_dist_2[129:150],evals_2[50][129:150],"b-")
# ax.plot(k_dist_2[149:165],evals_2[49][149:165],"b-")
# ax.plot(k_dist_2[164:200],evals_2[48][164:200],"b-")

ax.set_xlim(k_dist_2[0],k_dist_2[-1])
ax.set_ylim(-6.5,6.5)
# ax.set_title("Nanoribbon Band Structure",fontsize=25)
ax.set_xlabel("k",fontsize=25)
ax.set_ylabel("E",fontsize=25)
ax.yaxis.set_ticks([-6,-4,-2,0,2,4,6])
ax.set_yticklabels((-6.0,-4.0,-2.0,0.0,2.0,4.0,6.0),fontsize=18)
ax.xaxis.set_ticks(k_node_2)
ax.set_xticklabels((r'$0$',r'$\pi/\sqrt{3}$',r'$2\pi/\sqrt{3}$'),fontsize=18)
fig.tight_layout()


# plot the entanglement energy
fig, ax = plt.subplots()
for n in range(N_en):
    ax.plot(ex_3,energy_set[:,n],"k-")
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

# plot the entanglement spectrum 
fig, ax = plt.subplots()
for n in range(N_en):
    ax.plot(ex_3,en_set[:,n],"k-")
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
ax.set_ylim(-1.5,1.5)
fig.set_size_inches(5, 5)
fig.tight_layout()

x1 = np.linspace(0, 2*pi, 21) 
y1 = np.linspace(0, 6*pi/(3.**0.5), 31) 
X, Y = np.meshgrid(x1, y1)

dz = 2.0*(np.cos(2*X)+np.cos((3**0.5)*Y+X)+np.cos((3**0.5)*Y-X))\
    -0.0*(np.cos(1.5*X+0.5*(3**0.5)*Y)-np.cos(-1.5*X+0.5*(3**0.5)*Y)-np.cos((3**0.5)*Y))\
    +2.0*(np.cos(X)+np.cos(0.5*(3**0.5)*Y+0.5*X)+np.cos(0.5*(3**0.5)*Y-0.5*X))+1.5
dx = np.sin(2.5*X)*np.sin(0.5*(3**0.5)*Y)-np.sin(2.0*X)*np.sin((3**0.5)*Y)+np.sin(0.5*X)*np.sin(1.5*(3**0.5)*Y)
dy = np.sin(2.5*X)*np.cos(0.5*(3**0.5)*Y)-np.sin(2.0*X)*np.cos((3**0.5)*Y)-np.sin(0.5*X)*np.cos(1.5*(3**0.5)*Y)
nor = (dx**2+dy**2+dz**2)**0.5
dz = dz/nor
dx = dx/nor
dy = dy/nor

fig, ax = plt.subplots() 
im = ax.imshow(dz, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(0, 2*pi, 0, 6*pi/(3.**0.5)))
ax.xaxis.set_ticks([0.0,pi,2*pi])
ax.set_xticklabels((r'$0$', r'$\pi$', r'$2\pi$'),fontsize=20)
ax.yaxis.set_ticks([0.0,2*pi/np.sqrt(3),4*pi/np.sqrt(3),6*pi/np.sqrt(3)])
ax.set_yticklabels((r'$0$', r'$2\pi/\sqrt{3}$',r'$4\pi/\sqrt{3}$',r'$6\pi/\sqrt{3}$'),fontsize=20)
cbar=fig.colorbar(im,ticks=[-1.0,-0.5,0.0,0.5,1.0],shrink=0.7,aspect=8)
cbar.ax.tick_params(labelsize=15)
for ia in range (21):
  for ja in range (31):
    ax.arrow(X[ja][ia],Y[ja][ia],dx[ja][ia]/4.,dy[ja][ia]/4.,width=0.016,head_width=0.09,color='black')

fig.tight_layout()
fig.savefig('n=1(3).pdf')
plt.show()
