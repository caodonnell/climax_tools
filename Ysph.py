import numpy as np
import colorcorner as cc
import chnmethods as cm
import os, glob, sys, time
import chnmethods as cm
from scipy.integrate import quad
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm as cmap
import ctypes
lib = ctypes.cdll.LoadLibrary('/Users/codonnell/Google Drive/research/madcows/Ysph_integrands.so')


#constants
sigmaT = 6.6525E-25 #cm^2
#me = 9.11E-28 #g
#cc = 3.E10 #cm/s
me = 511. #keV/c^2
H0 = 70 #km/s/Mpc
h = H0/100.
cH0 = 4283. #Mpc (this is the c/H0 ratio, with useful units)

#a couple of useful conversion factors
Mpc2cm = 3.1E24
arcmin2rad = 291.E-6
keV2erg = 1.602E-9

#cosmology
OmegaM = 0.3
OmegaL = 0.7
calcDA = lambda z: (cH0) * quad(c_dAIntegrand, 0, z)[0]/(1.+z)
Ez = lambda z: np.sqrt(OmegaM*(1.+z)**3 + OmegaL)
Hz = lambda z: H0*Ez(z)



#for the Ysph integration
Rin = 0.1 #Mpc
Rout = np.arange(0.5, 2.1, 0.1)
#Rout = np.arange(0.8, 2.1, 0.2)
#Rout = np.arange(1.0, 2.1, 0.5)
#Rout = [1.5]
xsmall = 1.E-20
#xbig = 1.E20


#handling the ctypes integrals
c_dAIntegrand = lib.dAIntegrand
c_dAIntegrand.argtypes = (ctypes.c_int, ctypes.c_double)
c_dAIntegrand.restype  = ctypes.c_double
c_pxIntegrand = lib.pxIntegrand
c_pxIntegrand.argtypes = (ctypes.c_int, ctypes.c_double)
c_pxIntegrand.restype  = ctypes.c_double
c_Yintegrand = lib.Yintegrand
c_Yintegrand.argtypes = (ctypes.c_int, ctypes.c_double)
c_Yintegrand.restype  = ctypes.c_double

def calcY(Sradio, thetaC, dA, R, rc, params):
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    I0 = params['I0']
    Yint = quad(c_Yintegrand, Rin/rc, R/rc, args=(alpha, beta, gamma))
    #return max(Sradio * dA**2 * thetaC**2 * Yint[0] / (2*I0[0]), xsmall)
    return Sradio * dA**2 * thetaC**2 * Yint[0] / (2*I0[0])


def getY(chn, labels, dA, R, params):
    Y = []
    thetaCindex = cm.indexInList(labels, 'cluster.thetaC*')[0]
    Sradioindex = cm.indexInList(labels, 'cluster.Sradio*')[0]
    for iic,c in enumerate(chn):
        thetaC = c[thetaCindex] * arcmin2rad
        #y0 = c[Sradioindex]
        rc = dA*thetaC
        yderive = calcY(c[Sradioindex], thetaC, dA, R, rc, params)
        Y.append(yderive*1.E5)
        #if iic == 0: print c, Y
    #print min(Y), max(Y)
    return cc.quantile(Y, [0.50, 0.16, 0.84])

def getR500(chn, labels, dA, c500):
    thetaCindex = cm.indexInList(labels, 'cluster.thetaC*')[0]
    tlist = cc.quantile(chn[:, thetaCindex], [0.16, 0.50, 0.84])
    #print tlist
    rlist = [(t*arcmin2rad) * dA * c500 for t in tlist]
    return rlist


def getColors(ncurves):
    start = 0.0
    stop = 1.0
    cm_subsection = np.linspace(start, stop, ncurves) 
    colors = [cmap.rainbow(x) for x in cm_subsection]
    return colors[::-1]



def calcYsph(chnlist, labels = None, datpath = None,
             plotname = None, title = None):

    colors = getColors(len(chnlist))
    if plotname is None: plotname = 'Ysph.pdf'
    xmin = 0.45
    xmax = 2.1
    
    with PdfPages(plotname) as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, cname  in enumerate(chnlist):
            chnfile = cname
            if chnfile[-4:] != '.chn': chnfile = chnfile+'.chn'
            color = colors[i]
            chn = cm.getChn(chnfile)
            chnlabels = cm.getChnLabels(chnfile)
            z = cm.getz(chnfile)
            dA = calcDA(z)
            Ylist = []
            Yerrm = []
            Yerrp = []

            params = cm.getClusterParams(chnfile)
            if params['type'] == 'arnaud':
                print 'Using an arnaud model'
            else:
                print 'Using a GNFW model with [a,b,g] = [{:.2f},{:.2f},{:.2f}]'.format(params['alpha'], params['beta'], params['gamma'])
            I0 = quad(c_pxIntegrand, xsmall, np.inf,
                      args=(params['alpha'], params['beta'],
                            params['gamma']))
            params['I0'] = I0
        
            for r in Rout:
                print cname, r
                [y50, y16, y84] = getY(chn, chnlabels, dA, r, params)
                #print y50, y16, y84
                Ylist.append(y50)
                Yerrm.append(y50-y16)
                Yerrp.append(y84-y50)
                #print r, Ylist[-1], Yerrm[-1], Yerrp[-1]

            if datpath is not None:
                outf = open(datpath+cname+'_Ysph.dat', 'w')
                outf.write('{}\t{}\t{}\t{}\n'.format('Rout[Mpc]',
                                                     'Ysph[10^-5 Mpc^2]',
                                                     'Ysph_m', 'Ysph_p'))
                for r, y, ym, yp in zip(Rout, Ylist, Yerrm, Yerrp):
                    outf.write('{:.3f}\t\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(r,
                                                                         y,
                                                                         ym,
                                                                         yp))
                outf.close()

            #for i, r in enumerate(Rout):
            #    yerrm = Yerrm[i]
            #    yerrp = Yerrp[i]
            #    plt.errorbar(r, Ylist[i], fmt='o', c=color,
            #                 yerr = np.array([[yerrm, yerrp]]).T)
            #plt.errorbar([],[], fmt='o', c=color, label=cname)


            #label = cname
            if labels is None: label = cname
            else: label = labels[i]
            plt.errorbar(Rout, Ylist, fmt='o-', c=color,
                         yerr=[Yerrm, Yerrp], label=label)

            #r = getR500(chn, chnlabels, dA, params['c500'])
            #print r
            #plt.axvline(x=r[1])
            #plt.axvline(x=r[1], ls='dashed', color=color, lw=1)
            #rx = (r[0] - xmin)/(xmax-xmin)
            #rw = (r[2] - r[0])/(xmax-xmin)
            #rpatch = patches.Rectangle((rx, 0), width=rw, height=1,
            #                           color=color, alpha=0.3,
            #                           transform=ax.transAxes)
            #ax.add_patch(rpatch)
            

        #plt.xlabel(r'R$_{\mathrm{out}}$ [Mpc]')
        plt.xlabel(r'$R_{out}$ [Mpc]', fontsize=18)
        plt.xlim([xmin, xmax])
        #plt.ylabel(r'Y$_{\mathrm{sph}}$ [10$^{-5}$ Mpc$^2$]')
        plt.ylabel(r'$Y_{sph}$ [10$^{-5}$ Mpc$^2$]', fontsize=18)
        if title is not None:
            plt.title(title)
        
        legend = ax.legend(loc='upper left', numpoints=1, fontsize=18)
        legend.get_frame().set_facecolor('white')
        #legend.get_frame().set_edgecolor('white')

        pdf.savefig()
        plt.close()

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        #chnlist = [f.strip('.chn') for f in  glob.glob('W1106*/*chn')]
        #chnlist = ['stack_5', 'stack_7']
        #calcYsph(chnlist)

        #chnlist = ['stack_5', 'stack_5_recenter', 'stack_5_SZcenter']
        #labels = ['no shift', 'SZ-center shift', 'recalc center shift']
        #calcYsph(chnlist, labels=labels,
        #         title='stack of detected clusters')
        

        #chnlist = ['stack_lowz_lowRich', 'stack_highz_lowRich',
        #           'stack_lowz_highRich', 'stack_highz_highRich']
        #labels = ['low z, low richness', 'high z, low richness',
        #          'low z, high richness', 'high z, high richness']
        #calcYsph(chnlist, labels=labels, title='parameter quadrants')
        
        chnlist = ['shift_0', 'shift_5', 'shift_10', 'shift_30', 'shift_60']
        labels = []
        for c in chnlist:
            if c=='shift_0': labels.append('no shift')
            else:
                shift = c.split('_')[1]
                labels.append('shift$\sim \mathcal{N}(0,\,'+shift+'\'\'$ )')
        calcYsph(chnlist, labels=labels)#,
        #title='stack of 20 copies of MOO_1521+0452')
        

        #modelparams = np.loadtxt('modelparams.txt', usecols=[0,1,2,3])
        ##
        ###cmxsets = ['stack_5', 'W0037+33/W0037+33', 'W0123+25/W0123+25',
        ###           'W1231+65/W1231+65', 'W1521+04/W1521+04',
        ###           'W2231+11/W2231+11']
        ##
        #cmxsets = ['stack_5']
        #for cmxset in cmxsets:
        #    chnlist = [c.strip('.chn') for c in glob.glob(cmxset+'_model*.chn') if 'model4' not in c]
        #    chnlist.append('stack_5')
        #    #chnlist += [chnlist[0].split('_m')[0]]
        #    #chnlist = ['stack_5_model1', 'stack_5_modelArnaud',
        #    #'stack_5_modelSayers', 'stack_5']
        #    #labels = [c.split('_')[-1] for c in chnlist]
        #    labels = []
        #    for c in chnlist:
        #        #if 'model4' in c: continue
        #        if 'model' not in c: labels.append('Planck 2013')
        #        elif 'Arnaud' in c: labels.append('Arnaud+ 2010')
        #        elif 'Sayers' in c: labels.append('Sayers+ 2016')
        #        else:
        #            mnum = int(c[-1])
        #            if mnum == 0: labels.append('Bourdin+ 17 low-z')
        #            elif mnum ==1: labels.append('Bourdin+ 17 high-z')
        #            else: labels.append('({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(*modelparams[mnum]))
        #
        #    if '/' in cmxset: cname = cmxset.split('/')[0]
        #    else: cname = cmxset
        #    calcYsph(chnlist, labels=labels, plotname=cmxset+'_Ysph_short.pdf',
        #             title=cname+' $Y_{sph}$ for different models $(\\alpha,\\, \\beta,\\, \\gamma,\\, p_{0})$')

        #chnlist = ['stack_5_modelSayers', 'stack_5_modelSayers13']
        #calcYsph(chnlist)

        #print 'error'
        #chnlist = ['stack_lowz_highRich', 'stack_highz_highRich',
        #           'stack_lowz_highRich_nondetected',
        #           'stack_highz_highRich_nondetected']
        #labels = ['low z', 'high z',
        #          'low z (nondetected)', 'high z (nondetected)']
        #calcYsph(chnlist, labels=labels)
        
    else:
        chnlist = sys.argv[1:]
        #calcYsph(chnlist, datpath='./')
        calcYsph(chnlist, datpath = './')
    
    
    
