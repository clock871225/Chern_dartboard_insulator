from re import I
import numpy as np
import matplotlib.pyplot as plt

# plot nanoribbon band structures for all the CDIs (Fig. 3)
# load the data files

n1=np.load('../data/CDI_1_y_edge.npz')
k_dist_2_1=n1['arr_0']
k_node_2_1=n1['arr_1']
evals_2_1=n1['arr_2']

n1_2=np.load('../data/CDI_1_x_edge.npz')
k_dist_2_1_2=n1_2['arr_0']
k_node_2_1_2=n1_2['arr_1']
evals_2_1_2=n1_2['arr_2']

n2=np.load('../data/CDI_2_edge.npz')
k_dist_2_2=n2['arr_0']
k_node_2_2=n2['arr_1']
evals_2_2=n2['arr_2']

n4=np.load('../data/CDI_4_edge.npz')
k_dist_2_4=n4['arr_0']
k_node_2_4=n4['arr_1']
evals_2_4=n4['arr_2']

n3_1=np.load('../data/type1_CDI_3_y_edge.npz')
k_dist_2_3_1=n3_1['arr_0']
k_node_2_3_1=n3_1['arr_1']
evals_2_3_1=n3_1['arr_2']

n3_2=np.load('../data/type1_CDI_3_x_edge.npz')
k_dist_2_3_2=n3_2['arr_0']
k_node_2_3_2=n3_2['arr_1']
evals_2_3_2=n3_2['arr_2']

fig, ax = plt.subplots(3,2,figsize=(8,10)) 

# plot the band structures

for n in range(evals_2_2.shape[0]):
  ax[0][0].plot(k_dist_2_1,evals_2_1[n],"0.5")
  ax[0][1].plot(k_dist_2_1_2,evals_2_1_2[n],"0.5")
  ax[1][0].plot(k_dist_2_2,evals_2_2[n],"0.5")
  ax[1][1].plot(k_dist_2_4,evals_2_4[n],"0.5")
  ax[2][0].plot(k_dist_2_3_1,evals_2_3_1[n],"0.5")
  ax[2][1].plot(k_dist_2_3_2,evals_2_3_2[n],"0.5")

# highlight the gapless edge states with colors

ax[0][0].plot(k_dist_2_1[0:51],evals_2_1[49][0:51],"k-")
ax[0][0].plot(k_dist_2_1[50:151],evals_2_1[50][50:151],"k-")
ax[0][0].plot(k_dist_2_1[150:200],evals_2_1[49][150:200],"k-")
ax[0][0].plot(k_dist_2_1[0:51],evals_2_1[50][0:51],"b-")
ax[0][0].plot(k_dist_2_1[50:151],evals_2_1[49][50:151],"b-")
ax[0][0].plot(k_dist_2_1[150:200],evals_2_1[50][150:200],"b-")

ax[0][0].set_xlim(k_dist_2_1[0],k_dist_2_1[-1])
ax[0][0].set_ylim(-1.2,1.2)
ax[0][0].set_xlabel(r'$k_y$',fontsize=25)
ax[0][0].set_ylabel("E",fontsize=25)
ax[0][0].yaxis.set_ticks([-1.0,-0.5,0.0,0.5,1.0])
ax[0][0].set_yticklabels((-1.0,-0.5,0.0,0.5,1.0),fontsize=18)
ax[0][0].xaxis.set_ticks(k_node_2_1)
ax[0][0].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)

ax[0][1].plot(k_dist_2_1_2,evals_2_1_2[49],"r-")
ax[0][1].plot(k_dist_2_1_2,evals_2_1_2[50],"r-")

ax[0][1].set_xlim(k_dist_2_1_2[0],k_dist_2_1_2[-1])
ax[0][1].set_ylim(-1.2,1.2)
ax[0][1].set_xlabel(r'$k_x$',fontsize=25)
ax[0][1].set_ylabel("E",fontsize=25)
ax[0][1].yaxis.set_ticks([-1.0,-0.5,0.0,0.5,1.0])
ax[0][1].set_yticklabels((-1.0,-0.5,0.0,0.5,1.0),fontsize=18)
ax[0][1].xaxis.set_ticks(k_node_2_1_2)
ax[0][1].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)

ax[1][0].plot(k_dist_2_2[13:87],evals_2_2[50][13:87],"r-")
ax[1][0].plot(k_dist_2_2[13:87],evals_2_2[49][13:87],"r-")
ax[1][0].plot(k_dist_2_2[113:188],evals_2_2[50][113:188],"r-")
ax[1][0].plot(k_dist_2_2[113:188],evals_2_2[49][113:188],"r-")
ax[1][0].plot(k_dist_2_2[0:14],evals_2_2[50][0:14],"0.5")
ax[1][0].plot(k_dist_2_2[0:14],evals_2_2[49][0:14],"0.5")
ax[1][0].plot(k_dist_2_2[187:200],evals_2_2[50][187:200],"0.5")
ax[1][0].plot(k_dist_2_2[187:200],evals_2_2[49][187:200],"0.5")
ax[1][0].plot(k_dist_2_2[86:114],evals_2_2[50][86:114],"0.5")
ax[1][0].plot(k_dist_2_2[86:114],evals_2_2[49][86:114],"0.5")

ax[1][0].plot(k_dist_2_2[13:87],evals_2_2[51][13:87],"r-")
ax[1][0].plot(k_dist_2_2[13:87],evals_2_2[48][13:87],"r-")
ax[1][0].plot(k_dist_2_2[113:188],evals_2_2[51][113:188],"r-")
ax[1][0].plot(k_dist_2_2[113:188],evals_2_2[48][113:188],"r-")
ax[1][0].plot(k_dist_2_2[0:14],evals_2_2[51][0:14],"0.5")
ax[1][0].plot(k_dist_2_2[0:14],evals_2_2[48][0:14],"0.5")
ax[1][0].plot(k_dist_2_2[187:200],evals_2_2[51][187:200],"0.5")
ax[1][0].plot(k_dist_2_2[187:200],evals_2_2[48][187:200],"0.5")
ax[1][0].plot(k_dist_2_2[86:114],evals_2_2[51][86:114],"0.5")
ax[1][0].plot(k_dist_2_2[86:114],evals_2_2[48][86:114],"0.5")

ax[1][0].set_xlim(k_dist_2_2[0],k_dist_2_2[-1])
ax[1][0].set_ylim(-3.5,3.5)
ax[1][0].set_xlabel(r'$k_x$',fontsize=25)
ax[1][0].set_ylabel("E",fontsize=25)
ax[1][0].yaxis.set_ticks([-3,-2,-1,0,1,2,3])
ax[1][0].set_yticklabels((-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0),fontsize=18)
ax[1][0].xaxis.set_ticks(k_node_2_2)
ax[1][0].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)


ax[1][1].plot(k_dist_2_4[12:41],evals_2_4[50][12:41],"r-")
ax[1][1].plot(k_dist_2_4[12:41],evals_2_4[49][12:41],"r-")
ax[1][1].plot(k_dist_2_4[60:89],evals_2_4[50][60:89],"r-")
ax[1][1].plot(k_dist_2_4[60:89],evals_2_4[49][60:89],"r-")
ax[1][1].plot(k_dist_2_4[112:141],evals_2_4[50][112:141],"r-")
ax[1][1].plot(k_dist_2_4[112:141],evals_2_4[49][112:141],"r-")
ax[1][1].plot(k_dist_2_4[160:189],evals_2_4[50][160:189],"r-")
ax[1][1].plot(k_dist_2_4[160:189],evals_2_4[49][160:189],"r-")
ax[1][1].plot(k_dist_2_4[0:13],evals_2_4[50][0:13],"0.5")
ax[1][1].plot(k_dist_2_4[0:13],evals_2_4[49][0:13],"0.5")
ax[1][1].plot(k_dist_2_4[40:61],evals_2_4[50][40:61],"0.5")
ax[1][1].plot(k_dist_2_4[40:61],evals_2_4[49][40:61],"0.5")
ax[1][1].plot(k_dist_2_4[88:113],evals_2_4[50][88:113],"0.5")
ax[1][1].plot(k_dist_2_4[88:113],evals_2_4[49][88:113],"0.5")
ax[1][1].plot(k_dist_2_4[140:161],evals_2_4[50][140:161],"0.5")
ax[1][1].plot(k_dist_2_4[140:161],evals_2_4[49][140:161],"0.5")
ax[1][1].plot(k_dist_2_4[188:200],evals_2_4[50][188:200],"0.5")
ax[1][1].plot(k_dist_2_4[188:200],evals_2_4[49][188:200],"0.5")

ax[1][1].plot(k_dist_2_4[12:41],evals_2_4[51][12:41],"r-")
ax[1][1].plot(k_dist_2_4[12:41],evals_2_4[48][12:41],"r-")
ax[1][1].plot(k_dist_2_4[60:89],evals_2_4[51][60:89],"r-")
ax[1][1].plot(k_dist_2_4[60:89],evals_2_4[48][60:89],"r-")
ax[1][1].plot(k_dist_2_4[112:141],evals_2_4[51][112:141],"r-")
ax[1][1].plot(k_dist_2_4[112:141],evals_2_4[48][112:141],"r-")
ax[1][1].plot(k_dist_2_4[160:189],evals_2_4[51][160:189],"r-")
ax[1][1].plot(k_dist_2_4[160:189],evals_2_4[48][160:189],"r-")
ax[1][1].plot(k_dist_2_4[0:13],evals_2_4[51][0:13],"0.5")
ax[1][1].plot(k_dist_2_4[0:13],evals_2_4[48][0:13],"0.5")
ax[1][1].plot(k_dist_2_4[40:61],evals_2_4[51][40:61],"0.5")
ax[1][1].plot(k_dist_2_4[40:61],evals_2_4[48][40:61],"0.5")
ax[1][1].plot(k_dist_2_4[88:113],evals_2_4[51][88:113],"0.5")
ax[1][1].plot(k_dist_2_4[88:113],evals_2_4[48][88:113],"0.5")
ax[1][1].plot(k_dist_2_4[140:161],evals_2_4[51][140:161],"0.5")
ax[1][1].plot(k_dist_2_4[140:161],evals_2_4[48][140:161],"0.5")
ax[1][1].plot(k_dist_2_4[188:200],evals_2_4[51][188:200],"0.5")
ax[1][1].plot(k_dist_2_4[188:200],evals_2_4[48][188:200],"0.5")


ax[1][1].set_xlim(k_dist_2_4[0],k_dist_2_4[-1])
ax[1][1].set_ylim(-4.5,4.5)
ax[1][1].set_xlabel(r'$k_x$',fontsize=25)
ax[1][1].set_ylabel("E",fontsize=25)
ax[1][1].yaxis.set_ticks([-4,-2,0,2,4])
ax[1][1].set_yticklabels((-4.0,-2.0,0.0,2.0,4.0),fontsize=18)
ax[1][1].xaxis.set_ticks(k_node_2_4)
ax[1][1].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)

ax[2][0].plot(k_dist_2_3_1[34:167],evals_2_3_1[48][34:167],"r-")
ax[2][0].plot(k_dist_2_3_1[34:167],evals_2_3_1[49][34:167],"r-")
ax[2][0].plot(k_dist_2_3_1[34:167],evals_2_3_1[50][34:167],"r-")
ax[2][0].plot(k_dist_2_3_1[34:167],evals_2_3_1[51][34:167],"r-")

ax[2][0].plot(k_dist_2_3_1[0:35],evals_2_3_1[51][0:35],"0.5")
ax[2][0].plot(k_dist_2_3_1[0:35],evals_2_3_1[50][0:35],"0.5")
ax[2][0].plot(k_dist_2_3_1[0:35],evals_2_3_1[49][0:35],"0.5")
ax[2][0].plot(k_dist_2_3_1[0:35],evals_2_3_1[48][0:35],"0.5")

ax[2][0].plot(k_dist_2_3_1[67:134],evals_2_3_1[47][67:134],"k-")
ax[2][0].plot(k_dist_2_3_1[48:68],evals_2_3_1[49][48:68],"k-")
ax[2][0].plot(k_dist_2_3_1[133:153],evals_2_3_1[49][133:153],"k-")
ax[2][0].plot(k_dist_2_3_1[14:49],evals_2_3_1[50][14:49],"k-")
ax[2][0].plot(k_dist_2_3_1[0:15],evals_2_3_1[50][0:15],"0.5")
ax[2][0].plot(k_dist_2_3_1[152:187],evals_2_3_1[50][152:187],"k-")
ax[2][0].plot(k_dist_2_3_1[186:200],evals_2_3_1[50][186:200],"0.5")
ax[2][0].plot(k_dist_2_3_1[67:134],evals_2_3_1[52][67:134],"b-")
ax[2][0].plot(k_dist_2_3_1[48:68],evals_2_3_1[50][48:68],"b-")
ax[2][0].plot(k_dist_2_3_1[133:153],evals_2_3_1[50][133:153],"b-")
ax[2][0].plot(k_dist_2_3_1[14:49],evals_2_3_1[49][14:49],"b-")
ax[2][0].plot(k_dist_2_3_1[0:15],evals_2_3_1[49][0:15],"0.5")
ax[2][0].plot(k_dist_2_3_1[152:187],evals_2_3_1[49][152:187],"b-")
ax[2][0].plot(k_dist_2_3_1[186:200],evals_2_3_1[49][186:200],"0.5")

ax[2][0].set_xlim(k_dist_2_3_1[0],k_dist_2_3_1[-1])
ax[2][0].set_ylim(-4.5,4.5)
ax[2][0].set_xlabel(r'$k_y$',fontsize=25)
ax[2][0].set_ylabel("E",fontsize=25)
ax[2][0].yaxis.set_ticks([-4,-2,0,2,4])
ax[2][0].set_yticklabels((-4.0,-2.0,0.0,2.0,4.0),fontsize=18)
ax[2][0].xaxis.set_ticks(k_node_2_3_1)
ax[2][0].set_xticklabels((r'$0$',r'$\pi/\sqrt{3}$',r'$2\pi/\sqrt{3}$'),fontsize=18)

ax[2][1].plot(k_dist_2_3_2,evals_2_3_2[49],"r-")
ax[2][1].plot(k_dist_2_3_2,evals_2_3_2[50],"r-")

ax[2][1].set_xlim(k_dist_2_3_2[0],k_dist_2_3_2[-1])
ax[2][1].set_ylim(-4.5,4.5)
ax[2][1].set_xlabel(r'$k_x$',fontsize=25)
ax[2][1].set_ylabel("E",fontsize=25)
ax[2][1].yaxis.set_ticks([-4,-2,0,2,4])
ax[2][1].set_yticklabels((-4.0,-2.0,0.0,2.0,4.0),fontsize=18)
ax[2][1].xaxis.set_ticks(k_node_2_3_2)
ax[2][1].set_xticklabels((r'$0$',r'$\pi$',r'$2\pi$'),fontsize=18)

# labels

ax[0][0].annotate('a', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[0][1].annotate('b', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[1][0].annotate('c', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[1][1].annotate('d', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[2][0].annotate('e', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)
ax[2][1].annotate('f', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=16,fontname='arial',fontfamily='sans-serif',fontweight=600)

fig.tight_layout()

fig.savefig('edge.pdf')