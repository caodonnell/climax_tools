import numpy as np
import colorcorner as cc
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
c_kms = 3.E5 #km/s
me = 511. #keV/c^2
H0 = 70.2 #km/s/Mpc
h = H0/100.
#cH0 = 4283. #Mpc (this is the c/H0 ratio, with useful units)
cH0 = c_kms / H0
G = 430371. #km^2 Mpc / (10^14 Msolar s^2)
xsmall = 1.E-20
tolerance = 0.01
niter_max = 30

#a couple of useful conversion factors
Mpc2cm = 3.086E24
arcmin2rad = 291.E-6
keV2erg = 1.602E-9
SZAtoAndersson = 6.15001e18 #Mpc^2 --> Msun*keV

#cosmology
OmegaM = 0.274
OmegaL = 1.0 - OmegaM
calcDA = lambda z: (cH0) * quad(c_dAIntegrand, 0, z)[0]/(1.+z)
Ez = lambda z: np.sqrt(OmegaM*(1.+z)**3 + OmegaL)
Hz = lambda z: H0*Ez(z)
rho_c = lambda z: 3 * Hz(z) * Hz(z) / (8 * np.pi * G) #10^14 Msolar/Mpc^3


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

##Arnaud model
#alpha = 1.0510
#beta = 5.4905
#gamma = 0.3081
#c500 = 1.177
#print 'Using an Arnaud GNFW model'

##Planck model
#gamma = 0.31
#beta = 4.13
#alpha = 1.33
#c500 = 1.81
#print 'Using a Planck GNFW model'

#I0 = quad(c_pxIntegrand, xsmall, np.inf, args=(alpha, beta, gamma))
Rin = 0.01 #Mpc

#returns R500 in Mpc
def R500_from_M500(M500, rhocz, z):
    return ((3.0 * M500 / (4.0 * np.pi * rhocz * 500.0))**(1.0/3.0))

def scalerel_params(scalerel):
    params = {}
    if scalerel == 'andersson':
        params['A'] = np.random.normal(14.06,0.1,1)[0]
        params['B'] = np.random.normal(1.67,0.29,1)[0]
    elif scalerel == 'rozo':
        params['a'] = np.random.normal(0.87,0.18,1)[0]
        params['alpha'] = np.random.normal(1.71,0.08,1)[0]
        params['sigma'] = np.random.normal(0.15,0.02,1)[0]
    return params

#returns M500 in 10^14 Msolar
def Andersson_M500_from_Y(Y, z):
    Y = Y * SZAtoAndersson
    params = scalerel_params('andersson')
    A = params['A']
    B = params['B']
    E23 = Ez(z) ** (2./3)
    slope = 10.**(A)
    Mpivot = 3.0
    return Mpivot*((Y/(slope * E23))**(1./B))

def Rozo_M500_from_Y(Y):
    #print Y
    Y = Y * 1.E5
    params = scalerel_params('rozo')
    a = params['a']
    alpha = params['alpha']
    sigma = params['sigma']
    Mpivot = 4.4
    return np.exp((np.log(Y - 0.5*sigma**2) - a)/alpha)*Mpivot#+0.5*sigma**2


def calcY(Sradio, thetaC, dA, Rout, rc, params):
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    I0 = params['I0']
    Yint = quad(c_Yintegrand, Rin/rc, Rout/rc, args=(alpha, beta, gamma))
    return max(Sradio * dA**2 * thetaC**2 * Yint[0] / (2*I0[0]), xsmall)

    
#assumes thetaC is in radians already and Sradio is in comptonY
def iterateStep(Sradio, thetaC, dA, rhocz, z, params, scalerel):
    rc = dA*thetaC
    Rout = rc * params['c500']
    #scaleparams = scalerel_params(scalerel)
    Y_new = calcY(Sradio, thetaC, dA, Rout, rc, params)
    Y_old = -10
    niter = 0
    #while np.abs(R500 - Rout)/R500 > tolerance:
    while np.abs(Y_new - Y_old)/Y_new > tolerance and niter < niter_max:
        if scalerel=='andersson':
            M500 = Andersson_M500_from_Y(Y_new, z)
        elif scalerel=='rozo':
            M500 = Rozo_M500_from_Y(Y_new)
        else:
            print 'wrong scale rel? trying to use '+scalerel
            break
        R500 = R500_from_M500(M500, rhocz, z)
        Y_old = Y_new
        Rout = R500
        Y_new = calcY(Sradio, thetaC, dA, Rout, rc, params)
        niter += 1
    if niter <= 30:# and Y_new > 10*xsmall:
        return R500, Y_new, M500, niter
    else:
        return -1, -1, -1, niter

def getParamVals(param):
    q50, q16, q84 = cc.quantile(param, [0.50, 0.16, 0.84])
    errm = q50-q16
    errp = q84-q50
    return q50, errp, errm
    
#returns a string with 'val + unc - unc'
def getParamString(param):
    q50, q16, q84 = cc.quantile(param, [0.50, 0.16, 0.84])
    errm = q50-q16
    errp = q84-q50
    #err = ' \pm {:.3f}'.format((errm+errp)/2)
    err = '^{{+{:.3f}}}_{{-{:.3f}}}$'.format(errp, errm)
    return '${:.3f}'.format(q50)+err

#returns a string with 'val + unc - unc'
def makeParamString(q50, errp, errm):
    #err = ' \pm {:.3f}'.format((errm+errp)/2)
    err = '^{{+{:.3f}}}_{{-{:.3f}}}$'.format(errp, errm)
    return '${:.3f}'.format(q50)+err

#takes a .chn file
def main(chnfile, scalerel):
    print ''
    print '----------------------'
    print chnfile
    chn = cm.getChn(chnfile)
    chnLabels = cm.getChnLabels(chnfile)
    z = cm.getz(chnfile)

    params = cm.getClusterParams(chnfile)
    if params['c500'] < 0.0: params['c500'] = 1.81 #planck value

    if params['type'] == 'arnaud':
        print 'Using an arnaud model'
    else:
        print 'Using a GNFW model with [a,b,g] = [{:.2f},{:.2f},{:.2f}]'.format(params['alpha'], params['beta'], params['gamma'])

    I0 = quad(c_pxIntegrand, xsmall, np.inf,
              args=(params['alpha'], params['beta'], params['gamma']))
    params['I0'] = I0

    SradioIndex = cm.indexInList(chnLabels, 'cluster.Sradio*')[0]
    thetaCIndex = cm.indexInList(chnLabels, 'cluster.thetaC*')[0]
    dA = calcDA(z)
    #print Hz(z)
    rhocz = rho_c(z)
    #print rhocz

    Y = np.zeros(len(chn))
    R = np.zeros(len(chn))
    M = np.zeros(len(chn))
    N = np.zeros(len(chn))
    
    for iic, c in enumerate(chn):
        #if iic>2: break
        if iic%10000 == 0: print 'step {:d}'.format(iic)
        r, y, m, n = iterateStep(c[SradioIndex], c[thetaCIndex]*arcmin2rad,
                                 dA, rhocz, z, params, scalerel)
        if r == -1:
            print 'did not converge for step {:d}'.format(iic)
        
        R[iic] = r
        Y[iic] = y
        M[iic] = m
        N[iic] = n
        

    #Ymed = np.median(Y)
    #Mmed = np.median(M)
    #print Ymed, Mmed, M500_from_Y(Ymed, 14.06, 1.67, z)

    Y = Y[np.where(Y>0)]
    R = R[np.where(R>0)]
    M = M[np.where(M>0)]
    
    #make a quick histogram of niter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(N, bins=20, color='c')
    plt.xlabel('Number of iterations')
    avgstr = 'average: {:d}'.format(int(np.round(np.mean(N))))
    ax.text(0.8, 0.9, avgstr, transform=ax.transAxes)
    plt.savefig(chnfile.replace('.chn','')+'_'+scalerel+'_ysph_niter.pdf')
    plt.close()
    
    #make Y in prettier units
    Y = [y*1.E5 for y in Y]

    Ymid, Yerrp, Yerrm = getParamVals(Y)
    Rmid, Rerrp, Rerrm = getParamVals(R)
    Mmid, Merrp, Merrm = getParamVals(M)

    print 'R500 (Mpc):         '+getParamString(R)
    print 'Y500 (10^-5 Mpc^2): '+getParamString(Y)
    print 'M500 (10^14 Msun):  '+getParamString(M)
    
    outstr = '{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(Rmid, Rerrp, Rerrm, Ymid, Yerrp, Yerrm, Mmid, Merrp, Merrm)
    
    return outstr


if __name__ == "__main__":
    scalerel = 'andersson'
    
    if len(sys.argv) == 1:
        print 'needs a chnfile'
    else:
        chnlist = sys.argv[1:]
        outfile = open('Y500fit_'+scalerel+'_test.txt', 'w')
        outfile.write('#cluster R500 Rerr_p Rerr_m Y500 Yerr_p Yerr_m M500 Merr_p Merr_m\n')
        for chn in chnlist:
            if chn[-4:]=='.chn': 
                outstr = main(chn, scalerel=scalerel)
            else:
                outstr = main(chn+'.chn', scalerel=scalerel)

            cname = os.path.basename(chn).replace('.chn', '').replace('stack_', '').replace('_modelArnaud', '')
            outfile.write(cname+' '+outstr+'\n')
        outfile.close()



