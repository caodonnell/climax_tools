import numpy as np
import colorcorner as cc
import chnmethods as cm
import os, glob, sys, time
#sys.path.append('/Users/codonnell/Google Drive/research/madcows/')
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
    #return cc.quantile(Y, [0.50, 0.16, 0.84])
    return Y



def getColors(ncurves):
    start = 0.0
    stop = 1.0
    cm_subsection = np.linspace(start, stop, ncurves) 
    colors = [cmap.rainbow(x) for x in cm_subsection]
    return colors[::-1]



def calcYsph(chnlist, labels = None, datpath = None,
             plotname = None, title = None):

    if plotname is None: plotname = 'Ysph_dist.pdf'
       
    with PdfPages(plotname+'.pdf') as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #colors = ['orange', 'purple']
        colors=['blue']

        for i, cname  in enumerate(chnlist):
            chnfile = cname
            if chnfile[-4:] != '.chn': chnfile = chnfile+'.chn'

            chn = cm.getChn(chnfile)
            chnlabels = cm.getChnLabels(chnfile)
            z = cm.getz(chnfile)
            dA = calcDA(z)

            params = cm.getClusterParams(chnfile)
            if params['type'] == 'arnaud':
                print 'Using an arnaud model'
            else:
                print 'Using a GNFW model with [a,b,g] = [{:.2f},{:.2f},{:.2f}]'.format(params['alpha'], params['beta'], params['gamma'])
            I0 = quad(c_pxIntegrand, xsmall, np.inf,
                      args=(params['alpha'], params['beta'],
                            params['gamma']))
            params['I0'] = I0
        
            Y = getY(chn, chnlabels, dA, 1.0, params)
            print cc.quantile(Y, [0.50, 0.16, 0.84])

            #label = cname
            if labels is None: label = cname
            else: label = labels[i]
            plt.hist(Y, bins=30, color=colors[i], normed=True,
                     histtype='step', linewidth=2)
            plt.plot([],[],color=colors[i], label=label, lw=2)

        plt.xlabel("Y$_{sph}$(1.0 Mpc) [10$^{-5}$ Mpc$^2$]", fontsize=14)
        #plt.legend()
        if title is not None: plt.title(title)
        plt.xlim([-1.5,3])
        plt.axvline(x=0, c='k', ls='dashed', lw=2)
        #plt.tight_layout()
        pdf.savefig()
        plt.close()

    return 0


if __name__ == "__main__":
    #calcYsph(['stack_lowz_lowRich', 'stack_highz_lowRich'], labels=['low z', 'high z'], plotname='lowRich_Yhist', title='Low richness stacks')
    calcYsph(['stack_154_v4'], labels=['low z', 'high z'], plotname='stack_154_Yhist', title='Stack of all clusters')
