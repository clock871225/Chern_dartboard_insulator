from re import I
from pythtb import * # import TB model class
import numpy as np
from scipy.linalg import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys
sys.path.append('../')
from function.constant import *

# plot pseudospin textures for all the CDIs (Fig. 2)

# type I CDI_1

x1 = np.linspace(-pi, pi, 21) # spacing
y1 = np.linspace(-pi, pi, 21)  
X1, Y1 = np.meshgrid(x1, y1) # mesh
dz1 = np.cos(Y1) # pseudospin in z direction
dx1 = np.sin(Y1)*np.cos(X1) # pseudospin in x direction
dy1 = np.sin(Y1)*np.sin(X1) # pseudospin in y direction

# normalize the spin magnitude to 1
nor = (dx1**2+dy1**2+dz1**2)**0.5 
dz1 = dz1/nor
dx1 = dx1/nor
dy1 = dy1/nor

# type II CDI_1

x1 = np.linspace(-pi, pi, 21) # spacing
y1 = np.linspace(-pi, pi, 21) 
X1_2, Y1_2 = np.meshgrid(x1, y1) # mesh
dz1_2 = 0.5*(1+np.cos(X1_2))*(np.cos(2*Y1_2)-1)+1 # pseudospin in z direction
dx1_2 = 0.5*(1+np.cos(X1_2))*np.sin(2*Y1_2) # pseudospin in x direction
dy1_2 = np.sin(X1_2)*np.sin(Y1_2) # pseudospin in y direction

# normalize the spin magnitude to 1
nor = (dx1_2**2+dy1_2**2+dz1_2**2)**0.5
dz1_2 = dz1_2/nor
dx1_2 = dx1_2/nor
dy1_2 = dy1_2/nor

# CDI_2

x1 = np.linspace(-pi, pi, 21) # spacing
y1 = np.linspace(-pi, pi, 21) 
X2, Y2 = np.meshgrid(x1, y1) # mesh

dz2 = 1.0+np.cos(2*X2)+np.cos(2*Y2) # pseudospin in z direction
dx2 = -np.sin(X2)*np.sin(2*Y2) # pseudospin in x direction
dy2 = np.sin(2*X2)*np.sin(Y2) # pseudospin in y direction

# normalize the spin magnitude to 1
nor = (dx2**2+dy2**2+dz2**2)**0.5
dz2 = dz2/nor
dx2 = dx2/nor
dy2 = dy2/nor

# type I CDI_3

x1 = np.linspace(-4*pi/3, 4*pi/3, 25) # spacing
y1 = np.linspace(-2*pi/(3.**0.5), 2*pi/(3.**0.5), 21) 
X3, Y3 = np.meshgrid(x1, y1) # mesh

dz3 = 2.0*(np.cos(1.5*X3+0.5*(3**0.5)*Y3)+np.cos(-1.5*X3+0.5*(3**0.5)*Y3)+np.cos((3**0.5)*Y3))\
    +np.cos(3.0*X3+(3**0.5)*Y3)+np.cos(-3.0*X3+(3**0.5)*Y3)+np.cos(2.0*(3**0.5)*Y3)+2.0 # pseudospin in z direction
dx3 = np.sin(2.5*X3)*np.sin(0.5*(3**0.5)*Y3)-np.sin(2.0*X3)*np.sin((3**0.5)*Y3)+np.sin(0.5*X3)*np.sin(1.5*(3**0.5)*Y3) # pseudospin in x direction
dy3 = -np.cos(2.5*X3)*np.sin(0.5*(3**0.5)*Y3)-np.cos(2.0*X3)*np.sin((3**0.5)*Y3)+np.cos(0.5*X3)*np.sin(1.5*(3**0.5)*Y3) # pseudospin in y direction

# normalize the spin magnitude to 1
nor = (dx3**2+dy3**2+dz3**2)**0.5
dz3 = dz3/nor
dx3 = dx3/nor
dy3 = dy3/nor

# type II CDI_3

# x1 = np.linspace(0, 2*pi, 21) # spacing
# y1 = np.linspace(0, 6*pi/(3.**0.5), 31) 
# X, Y = np.meshgrid(x1, y1) # mesh

# dz = 2.0*(np.cos(2*X)+np.cos((3**0.5)*Y+X)+np.cos((3**0.5)*Y-X))\
#     -0.0*(np.cos(1.5*X+0.5*(3**0.5)*Y)-np.cos(-1.5*X+0.5*(3**0.5)*Y)-np.cos((3**0.5)*Y))\
#     +2.0*(np.cos(X)+np.cos(0.5*(3**0.5)*Y+0.5*X)+np.cos(0.5*(3**0.5)*Y-0.5*X))+1.5
# dx = np.sin(2.5*X)*np.sin(0.5*(3**0.5)*Y)-np.sin(2.0*X)*np.sin((3**0.5)*Y)+np.sin(0.5*X)*np.sin(1.5*(3**0.5)*Y) # pseudospin in x direction
# dy = np.sin(2.5*X)*np.cos(0.5*(3**0.5)*Y)-np.sin(2.0*X)*np.cos((3**0.5)*Y)-np.sin(0.5*X)*np.cos(1.5*(3**0.5)*Y) # pseudospin in y direction

# normalize the spin magnitude to 1
# nor = (dx**2+dy**2+dz**2)**0.5
# dz = dz/nor
# dx = dx/nor
# dy = dy/nor

# CDI_4

x1 = np.linspace(-pi, pi, 31) # spacing
y1 = np.linspace(-pi, pi, 31)
X4, Y4 = np.meshgrid(x1, y1) # mesh
dz4 = 3.0+np.cos(2*X4)+np.cos(2*Y4)+np.cos(4*X4)+np.cos(4*Y4)+4.0*np.cos(X4)*np.cos(Y4) # pseudospin in z direction
dx4 = -np.sin(X4)*np.sin(4*Y4)+np.sin(4*X4)*np.sin(Y4) # pseudospin in x direction
dy4 = np.sin(2*X4)*np.sin(4*Y4)-np.sin(4*X4)*np.sin(2*Y4) # pseudospin in y direction

# normalize the spin magnitude to 1
nor = (dx4**2+dy4**2+dz4**2)**0.5
dz4 = dz4/nor
dx4 = dx4/nor
dy4 = dy4/nor

# zoom-in subplot for CDI_4

x1 = np.linspace(0, pi, 21) # spacing
y1 = np.linspace(0, pi, 21)
X42, Y42 = np.meshgrid(x1, y1) # mesh
dz42 = 3.0+np.cos(2*X42)+np.cos(2*Y42)+np.cos(4*X42)+np.cos(4*Y42)+4.0*np.cos(X42)*np.cos(Y42) # pseudospin in z direction
dx42 = -np.sin(X42)*np.sin(4*Y42)+np.sin(4*X42)*np.sin(Y42) # pseudospin in x direction
dy42 = np.sin(2*X42)*np.sin(4*Y42)-np.sin(4*X42)*np.sin(2*Y42) # pseudospin in y direction

# normalize the spin magnitude to 1
nor = (dx42**2+dy42**2+dz42**2)**0.5
dz42 = dz42/nor
dx42 = dx42/nor
dy42 = dy42/nor

fig, ax = plt.subplots(3,2,figsize=(8,10),gridspec_kw={'width_ratios': [1, 1]}) 

# plot the pseudopin textures in z direction with colors

im0 = ax[0][0].imshow(dz1, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(-pi, pi, -pi, pi))
im1 = ax[0][1].imshow(dz1_2, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(-pi, pi, -pi, pi))
im2 = ax[1][0].imshow(dz2, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(-pi, pi, -pi, pi))
im3 = ax[1][1].imshow(dz3, interpolation ='bilinear', origin ='lower', cmap ="viridis", extent =(-4*pi/3, 4*pi/3, -2*pi/(3.**0.5), 2*pi/(3.**0.5)))
im4 = ax[2][0].imshow(dz4, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(-pi, pi, -pi, pi))
im5 = ax[2][1].imshow(dz42, interpolation ='bilinear', origin ='lower', cmap ="viridis",  extent =(0, pi, 0.0, pi))
ax[0][0].xaxis.set_ticks([-pi,0.0,pi])
ax[0][0].set_xticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[0][0].yaxis.set_ticks([-pi,0.0,pi])
ax[0][0].set_yticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[0][1].xaxis.set_ticks([-pi,0.0,pi])
ax[0][1].set_xticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[0][1].yaxis.set_ticks([-pi,0.0,pi])
ax[0][1].set_yticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[1][0].xaxis.set_ticks([-pi,0.0,pi])
ax[1][0].set_xticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[1][0].yaxis.set_ticks([-pi,0.0,pi])
ax[1][0].set_yticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[1][1].xaxis.set_ticks([-4*pi/3,0.0,4*pi/3])
ax[1][1].set_xticklabels((r'$-4\pi/3$',r'$0$', r'$4\pi/3$'),fontsize=16)
ax[1][1].yaxis.set_ticks([-2*pi/np.sqrt(3),0.0,2*pi/np.sqrt(3)])
ax[1][1].set_yticklabels((r'$-2\pi/\sqrt{3}$',r'$0$', r'$2\pi/\sqrt{3}$'),fontsize=16)
ax[2][0].xaxis.set_ticks([-pi,0.0,pi])
ax[2][0].set_xticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[2][0].yaxis.set_ticks([-pi,0.0,pi])
ax[2][0].set_yticklabels((r'$-\pi$',r'$0$', r'$\pi$'),fontsize=16)
ax[2][1].xaxis.set_ticks([0.0,pi])
ax[2][1].set_xticklabels((r'$0$', r'$\pi$'),fontsize=16)
ax[2][1].yaxis.set_ticks([0.0,pi])
ax[2][1].set_yticklabels((r'$0$', r'$\pi$'),fontsize=16)

lines = [[(0, 0), (0, pi)], [(0, pi), (pi, pi)], [(pi, 0), (pi, pi)],[(0,0),(pi,0)]]
lc = LineCollection(lines, colors='r', linewidths=2)
ax[2][0].add_collection(lc)

# labels

ax[0][0].annotate('a', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[0][1].annotate('b', xy=(-0.1, 1.05), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[1][0].annotate('c', xy=(-0.1, 1.105), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[1][1].annotate('d', xy=(-0.1, 1.225), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[2][0].annotate('e', xy=(-0.1, 1.115), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[2][1].annotate('f', xy=(-0.1, 1.13), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[1][1].annotate(r'$n_z$', xy=(1.1, 1.4), xycoords='axes fraction', fontsize=24)
fig.subplots_adjust(left=0.5)

# colorbar

cbar=fig.colorbar(im0,ticks=[-1.0,-0.5,0.0,0.5,1.0],shrink=0.4,aspect=15,ax=ax[0:,:],location='right',pad=-0.42)
cbar.ax.tick_params(labelsize=15)

# plot the pseudopin textures in x and y direction with arrows

for ia in range (21):
  for ja in range (21):
    ax[0][0].arrow(X1[ja][ia],Y1[ja][ia],dx1[ja][ia]/5.,dy1[ja][ia]/5.,width=0.015,head_width=0.07,color='black')
    ax[0][1].arrow(X1_2[ja][ia],Y1_2[ja][ia],dx1_2[ja][ia]/5.,dy1_2[ja][ia]/5.,width=0.015,head_width=0.07,color='black')
    ax[1][0].arrow(X2[ja][ia],Y2[ja][ia],dx2[ja][ia]/5.,dy2[ja][ia]/5.,width=0.015,head_width=0.07,color='black')
    ax[2][1].arrow(X42[ja][ia],Y42[ja][ia],dx42[ja][ia]/9.,dy42[ja][ia]/9.,width=0.006,head_width=0.035,color='black')

for ia in range (31):
  for ja in range (31):
    ax[2][0].arrow(X4[ja][ia],Y4[ja][ia],dx4[ja][ia]/6.,dy4[ja][ia]/6.,width=0.001,head_width=0.033,color='black')

for ia in range (25):
  for ja in range (21):
    ax[1][1].arrow(X3[ja][ia],Y3[ja][ia],dx3[ja][ia]/5.,dy3[ja][ia]/5.,width=0.01,head_width=0.07,color='black') 

fig.tight_layout()
fig.savefig('pseudospin.pdf')
plt.show()