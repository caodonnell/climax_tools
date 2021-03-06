import glob, os, fnmatch, sys
import numpy as np
from astropy.io import fits
from astropy import wcs
from wcsaxes import WCSAxes
import colorcorner as cc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.patches import Rectangle, Circle
from PyPDF2 import PdfFileMerger
import aplpy as apy

small = 1.E-6

def ast2deg(ast, ra=False):
    if ra:
        asts = ast.split(':')
        return 15.*(float(asts[0])+float(asts[1])/60.0
                   +float(asts[2])/3600.0)
    else:
        sign = ast[0]
        asts = ast[1:].split(':')
        if sign=='+':
            return float(asts[0])+float(asts[1])/60.0+float(asts[2])/3600.0
        else:
            return -1.0*(float(asts[0])+float(asts[1])/60.0
                         +float(asts[2])/3600.0)


def deg2ast(deg, ra=False):
    factor = 1
    if ra: deg = deg/15.
    else:
        if deg < 0:
            factor = -1
            deg = deg*factor
    dh = int(deg)
    dr = (deg - dh)*60.
    mm = int(dr)
    ss = (dr - mm)*60.
    return '{}:{}:{:.2f}'.format(factor*dh, mm, ss)

#truncate color maps
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#returns indices of items in list that contain str but not badstr
#if single or multiple matches, returns a list
#if no matches, returns -1
def indexInList(lst, str, badstr=None):
    matches = []
    if badstr is None:
        matches = [i for i,j in enumerate(lst)
                   if fnmatch.fnmatch(j, str)]
    else:
        matches = [i for i,j in enumerate(lst)
                   if fnmatch.fnmatch(j, str) and bastr not in j]
    return matches
    #figure out what to return
    #if len(matches) > 0:
    #    return matches
    #else:
    #    return -1
    
def getChnLabels(chnfile):
    #get labels
    chn = open(chnfile, 'r')
    lines = chn.readlines()
    chn.close()
        
    start = 0 #get index of first column label
    for line in lines:
        if 'Columns are' in line:
            start += 3
            break
        else:
            start += 1

    #now get labels
    labels = []
    for i in range(start, len(lines)):
        #print lines[i]
        line = lines[i]
        if '//\n' == line: break
        ss = line.replace('//', '').split()
        #print ss
        #if ss[0] != '//' or len(ss)==1: break
        if len(ss)>3:
            labels.append(ss[1]+' '+ss[2])
        #elif len(ss) == 3:
        #    labels.append(ss[2])
        else:
            labels.append(ss[-1])
        #print ss, labels[-1]

    return labels

#deals with the fact the chn file generated by climax has a multiplicity
#factor for steps that get repeated
#this method assumes the last column is multiplicity and generates
#a chn array that has ALL steps included
def chnMults(chn):
    chnlen = len(chn[:,0])
    newchn = []
    for i in range(chnlen):
        for j in range(int(chn[i, -1])):
            newchn.append(list(chn[i,:]))
    return np.array(newchn)

#another way of getting a chain file and dealing with the multiplicities
def getChn(chnfile):
    chn = np.loadtxt(chnfile, comments='//')
    if len(chn) > 0:
        mults = np.loadtxt(chnfile, dtype='int', comments='//', usecols=[-1])
        #print sum(mults)
        newchn = []
        for c,m in zip(chn, mults):
            #c[2] = c[2]*1.E4
            #c[4] = c[4]*1.E5
            for j in range(m): newchn.append(c)
        print chnfile+': '+'{:.3f}%'.format(100.0*len(mults)/sum(mults))
    return np.array(newchn)
    

def get20cmSources(cluster):
    path = '/Users/codonnell/Google Drive/research/madcows/images/'
    sources = []

    try:
        nvss = np.loadtxt(path+cluster+'/'+cluster+'_nvss.txt',
                          comments='//')
        if len(np.shape(nvss)) > 1:
            for s in nvss:
                #print s
                #print s[0], s[0]*15.0
                sources.append((s[0]*15.0, s[2], 'nvss'))
        else:
            sources.append((nvss[0]*15.0, nvss[2], 'nvss'))
    except:
        pass

    try:
       first = np.loadtxt(path+cluster+'/'+cluster+'_first.txt',
                          comments='//') 

       if len(np.shape(first)) > 1:
           for s in first: sources.append((s[0]*15.0, s[2], 'first'))
       else:
           sources.append((first[0]*15.0, first[2], 'first'))
    except:
        pass
           
    #print sources
    return sources

def plot20cmSources(f, yw, sourcelist):
    width = 0.12/60./np.cos(np.deg2rad(yw))
    height = 0.65/60.
    for s in sourcelist:
        if s[2]=='nvss':
            #print s[0], s[1], ': ', deg2ast(s[0], ra=True), deg2ast(s[1], ra=False)
            x1 = s[0] #- width/2.
            y1 = s[1] - height
            x2 = s[0] - height
            y2 = s[1] - width/2.
            f.show_rectangles([x1,x2], [y1,y2],
                              [width, height], [height, width],
                              edgecolor='k', lw=0.01, facecolor='plum')
        if s[2]=='first':
            x1 = s[0] #- width/2.
            y1 = s[1] + height#-0.002
            x2 = s[0] + height#-0.002
            y2 = s[1] - width/2.
            f.show_rectangles([x1,x2], [y1,y2],
                              [width, -1.0*height], [-1.0*height, width],
                              edgecolor='k', lw=0.01, facecolor='lightpink')
    return


#to make it easier to account for all point source models
class chnModels:
    def __init__(self, name):
        self.xoff = None
        self.xoff_err = None
        self.yoff = None
        self.yoff_err = None
        self.ra = None
        self.ra_err = None
        self.dec = None
        self.dec_err = None
        self.Sradio = None
        self.Sradio_err = None
        self.name = name

#this assumes you have a list models of chnModel objects
#it searches for the one with a matching name and returns the index
def findChnModel(models, name):
    matches = []
    for i, m in enumerate(models):
        if m.name == name: matches.append(i)
    return matches

#searches for the value of a parameter to a chn model
#if the parameter was a float (i.e., fit by climax), it returns the value
#associated with the best-fit model
#otherwise, it returns the set value
def findParamValue(chn, labels, bestmodel, line):
    if ':' in line: #fit value
        ls = line.split()
        pname = ls[1]
        #print pname
        chnindex = indexInList(labels, pname+'*')[0]
        #print chnindex, chn[bestmodel, chnindex]
        #return chn[bestmodel, chnindex]
        q16, q50, q84 = cc.quantile(chn[:,chnindex], [0.16, 0.50, 0.84])
        #print q16, q50, q84
        return q50, max(q50-q16, q84-q50)
    else: #fixed value
        ls = line.split('=')
        value = ls[1].split()
        return float(value[0]), 0.
    return -1

#gets point sources from chn file
def getChnSources(chn, labels, chnfile):
    sources = []
    chnf = open(chnfile, 'r')
    lines = chnf.readlines()
    chnf.close()
    lnlike = chn[:, indexInList(labels, '*likelihood*')]
    #print lnlike, max(lnlike), indexInList(labels, '*likelihood*'), labels
    bestmodel = np.where(lnlike == max(lnlike))[0][0]
    models = []

    #look for pt source models
    for line in lines:
        if ('Model' in line or 'nvss' in line or 'first' in line) and '// //' not in line:
            if 'addmodel' in line:
                ls = line.split()
                indexname = indexInList(ls, 'name=*')[0]
                #print 'new: ', ls[indexname][5:]
                model = chnModels(ls[indexname][5:])
                models.append(model)
            elif '=' in line:
                #print line
                ls = line.split()
                mname = ls[1][0:ls[1].find('.')]
                #print mname, findChnModel(models, mname)
                modelindex = findChnModel(models, mname)[0]
                p, p_err = findParamValue(chn, labels, bestmodel, line)
                #print line
                if '.xoff' in line:
                    models[modelindex].xoff = p
                    models[modelindex].xoff_err = p_err
                if '.yoff' in line:
                    models[modelindex].yoff = p
                    models[modelindex].yoff_err = p_err
                if '.ra' in line:
                    #print models[modelindex].name, p, p_err
                    models[modelindex].ra = p
                    models[modelindex].ra_err = p_err
                    #print p, models[modelindex].ra
                if '.dec' in line:
                    models[modelindex].dec = p
                    models[modelindex].dec_err = p_err
                if '.Sradio' in line:
                    #print line, p, p_err
                    models[modelindex].Sradio = p
                    models[modelindex].Sradio_err = p_err
                #else: print line
    #print models[0].name, models[0].Sradio
    return models
    #for m in models:
    #    if m.xoff is not None:
    #        sources.append((m.name, m.xoff, m.yoff, 'off'))
    #    else:
    #        sources.append((m.name, m.ra, m.dec, 'radec'))
    ##print sources
    #return sources

#now plot the sources from the chn
def plotChnSources(f, fitsfile, sourcelist):
    header = fits.getheader(fitsfile)
    for s in sourcelist:
        if s.xoff is not None:
            c_ra = header['crval1']
            c_dec = header['crval2']
            ra = c_ra + (s.xoff/3600.)#*np.cos(np.deg2rad(c_dec))
            dec = s.yoff/3600.0 + c_dec
        else:
            #print s.name
            ra = s.ra
            dec = s.dec
        #print ra, dec
        #print s.name, s.ra, ra, dec
        f.show_circles(ra, dec, 0.0125, linewidth=0.5,
                       edgecolor='crimson', facecolor='none')
    return          

def plotClusterCircle(f, fitsfile, cname, recenterlist= None):
    header = fits.getheader(fitsfile)
    f.show_circles(header['crval1'], header['crval2'], 0.033,
                   linewidth=1.5, linestyle='dotted',
                   color='crimson')
    if recenterlist is not None:
        index = indexInList(recenterlist[:,0], cname)
        if len(index) == 1:
            index = index[0]
            ra = float(recenterlist[index,1])
            dec = float(recenterlist[index,2])
            f.show_circles(ra, dec, 0.033, linewidth=1.5,
                           linestyle='dotted', color='crimson')

def plotFittedClusterCircle(f, chn, labels, xw, yw):
    xoffindex = indexInList(labels, 'cluster.xoff*')
    yoffindex = indexInList(labels, 'cluster.yoff*')
    if len(xoffindex) == 1 and len(yoffindex)==1:
        xoff = cc.quantile(chn[:, xoffindex[0]], 0.5)[0]/60.
        yoff = cc.quantile(chn[:, yoffindex[0]], 0.5)[0]/60.
        #print xoff, yoff, xw, yw
        f.show_circles(xw+xoff, yw+yoff, 0.033,
                       linewidth = 1.5, linestyle='dotted',
                       color = 'crimson')
                       
        
            
#makes the plots for a madcows cluster
def makeplot(path, plotpath, cname, ptsrc = False, corner=True):
    print ''
    print '----------------------'
    #print path
    #cname = os.path.basename(os.path.dirname(path))
    #cname = 'ptsrc'
    shortdata = path+cname+'_data_short.fits'
    shortmodel = path+cname+'_model_short.fits'
    shortres = path+cname+'_res_short.fits'
    shortnoise = path+cname+'_noise_short.fits'
    longdata = path+cname+'_data_long.fits'
    longmodel = path+cname+'_model_long.fits'
    longres = path+cname+'_res_long.fits'
    longnoise = path+cname+'_noise_long.fits'
    chnfile = path+cname+'.chn'
    recenterlist = np.loadtxt('/Users/codonnell/Google Drive/research/madcows/madcows_radec_recentered.txt', usecols=[1,-3,-2], dtype='str')

    images = [shortdata, shortmodel, shortres, longdata, longmodel, longres]

    if os.path.isfile(chnfile):
        #chn = np.loadtxt(chnfile, comments='//')
        #mults = chn[:,-1]
        #chn = chnMults(chn)
        chn = getChn(chnfile)#[:100000]
        #chn = np.delete(chn, range(100000,150000), axis=0)
        labels = getChnLabels(chnfile)
        #print labels
        chnsources = getChnSources(chn, labels, chnfile)
        #because I know some of the orders of magnitude...
        #print indexInList(labels, 'cluster.Sradio*')
        if len(indexInList(labels, 'cluster.*')) > 0:
            clusterSradio_index = indexInList(labels, 'cluster.Sradio*')[0]
            Ysph_index = indexInList(labels, '*Ysph*')[0]
            #print chn
            chn[:,clusterSradio_index] = chn[:,clusterSradio_index]*1E4
            labels[clusterSradio_index] = 'cluster.Sradio ('+r'$10^{-4}$'+' y)'
            chn[:,Ysph_index] = chn[:,Ysph_index]*1E5
            labels[Ysph_index] = 'cluster.Ysph ('+r'$10^{-5}$'+' Mpc'+r'$^2$'+')'
    else:
        chn = []

    #if len(chn) > 0:
    #    print path+': '+'{:.3f}%'.format(100.0*len(mults)/sum(mults))
    #else:
    #    print path
    
    
    #print plotpath+'/'+cname+'_images.pdf'
    with PdfPages(plotpath+'/'+cname+'_images.pdf') as pdf:
        fig = plt.figure()
        mycmap = truncate_colormap(plt.get_cmap('viridis'), 0.0, 0.95)
        dx = 0.2
        dy = 0.35
    
        for ii,i in enumerate(images):
            if os.path.isfile(i):
                hdu = fits.open(i)
                dims = np.shape(hdu[0].data)
                hdu.close()
                
                xpos = 0.08 + (ii%3)*0.32
                ypos = 0.55
                if ii > 2: ypos = 0.1
                f = apy.FITSFigure(i, figure=fig,
                                   subplot=[xpos, ypos, dx, dy])
                #f = apy.FITSFigure(i, figure=fig, subplot=(2,3,ii+1))
                f.set_title(os.path.basename(i).replace('.fits',''),
                            fontsize=8)
                f.tick_labels.set_xformat('hh:mm:ss')
                f.tick_labels.set_yformat('dd:mm:ss')
                f.set_tick_labels_font(size=4)
                f.set_axis_labels_font(size=7)
                xw, yw = f.pixel2world(int(dims[0]/2), int(dims[1]/2))
                print xw, yw
            
                vmin = None
                vmax = None
                cbarname = 'SNR'
                if 'res' in os.path.basename(i):
                    #print os.path.basename(i)
                    vmin = -5.0
                    vmax = 5.0
                    cbarname = 'SNR (clipped)'
                f.show_colorscale(vmin=vmin, vmax=vmax, cmap=mycmap)

                if 'res_' in i: #add noise string
                    noisehdu = fits.open(i.replace('res_','noise_'))
                    noise = np.min(noisehdu[0].data[np.nonzero(noisehdu[0].data)])
                    noisehdu.close()
                    noisestr = 'noise: '+'{:.3f}'.format(noise*1000)+' mJy'
                    f.add_label(0.3,0.9, noisestr,
                                bbox=dict(facecolor='white',
                                          alpha=0.8,
                                          edgecolor='none'),
                                relative=True, size=5)

                #add 20cm and chain sources
                if ptsrc:
                    cmsources = get20cmSources(cname)
                    plot20cmSources(f, yw, cmsources)
                    if len(chn)>0: plotChnSources(f, i, chnsources)

                #add circle for the center
                #plotClusterCircle(f, i, cname)#, recenterlist)
                #xoff = -333.16428
                #yoff = -10.77763
                #f.show_circles([xw+(xoff/3600.)*np.cos(np.deg2rad(-9.74))],
                #               [yw+(yoff/3600.)], 0.033,
                #               color='crimson', lw=1.5)
                #plotFittedClusterCircle(f, chn, labels, xw, yw)
            
                f.add_colorbar(axis_label_text=cbarname)
                f.colorbar.set_width=0.05
                f.colorbar.set_font(size=6)
                f.colorbar.set_axis_label_font(size=6)
                
        fig.canvas.draw()
        pdf.savefig()
        plt.close()

    if corner and len(chn)>0:
        #print 'working on corner'
        clusterindex = indexInList(labels, 'cluster.*')
        ptsrcindex = [i for i in range(len(labels)-3)
                      if i not in clusterindex]
        if len(clusterindex) > 1:
            print 'corner plot for cluster'
            with PdfPages(plotpath+'/'+cname+'_cluster_corner.pdf') as pdf:
                fig = cc.corner(chn[:,clusterindex],
                                labels=np.array(labels)[clusterindex],
                                cornerTitle=cname, cornerSigmaTitle=None,
                                show_titles=True, quantiles=(0.16,0.84),
                                title_kwargs={"fontsize": 8},
                                label_kwargs={"fontsize": 10},
                                title_fmt='.3f', centerline=True,
                                histcolor='g', quantilescolor='fuchsia',
                                centerlinecolor='fuchsia',
                                centershade=True, centershadecolor='k',
                                bins=30, density_cmap='viridis_r',
                                plot_contours=False)
                #print 'made corner plot'
                pdf.savefig()
                plt.close()
        if len(ptsrcindex) > 0:
	    with PdfPages(plotpath+'/'+cname+'_ptsrc_corner.pdf') as pdf:
	        for mincol in range(0, len(ptsrcindex), 10):
	            maxcol = min(mincol+10, len(ptsrcindex))
	            print 'corner for ptsrc param {:d} to {:d}'.format(mincol, maxcol)
	            fig = cc.corner(chn[:,ptsrcindex[mincol:maxcol]],
	                            labels=np.array(labels)[ptsrcindex[mincol:maxcol]],
	                            cornerTitle=cname, cornerSigmaTitle=None,
	                            show_titles=True, quantiles=(0.16,0.84),
	                            title_kwargs={"fontsize": 8},
	                            label_kwargs={"fontsize": 10},
	                            title_fmt='.3f', centerline=True,
	                            histcolor='g', quantilescolor='fuchsia',
	                            centerlinecolor='fuchsia',
	                            centershade=True, centershadecolor='k',
	                            bins=30, density_cmap='viridis_r',
	                            plot_contours=False)
	            #print 'made corner plot'
	            pdf.savefig()
	        plt.close()
	
        #pdflist = [plotpath+cname+'_corner.pdf',
        #           plotpath+cname+'_images.pdf']
        #outfile = PdfFileMerger()
        #
        #for f in pdflist:
        #    outfile.append(open(f, 'rb'))
        #
        #outfile.write(open(plotpath+cname+'_all.pdf', 'wb'))

    return 0
    

if __name__ == "__main__":
    basepath = '/Users/codonnell/Google Drive/research/madcows/'
    #args = sys.argv
    #print sys.argv

    
    if len(sys.argv) > 1:
        stackpath = sys.argv[1]
    else:
        stackpath = './'
        #stackpath = basepath+'stacking/cleanimages/stack_from_data_uvfs/'
        #stackpath = basepath+'stacking/test/pixelsize/'
    
    if len(sys.argv) > 2:
        cname = sys.argv[2]
        #if len(sys.argv) == 4:
        #    ptsrc = sys.argv[3]
        #else:
        #    ptsrc = False
        makeplot(stackpath, stackpath, cname, ptsrc=False, corner=True)
    else:
        #cnamebase = 'stack_10_pt'
        #for i in range(1,16):
        #    if i==10: continue
        #    cname = cnamebase+str(i)
        #    makeplot(stackpath, stackpath, cname)
        #cname = 'stack_8'
        cnamebase = 'stack_5_'
        for cmx in glob.glob(cnamebase+'*.cmx'):
            cname = cmx.strip('.cmx')
            makeplot(stackpath, stackpath, cname)
    
    

    #paths = glob.glob(basepath+'stacking/ptsrc_removal/ptsrc_removed/W*/')
    #for path in paths:
    #    cname = os.path.basename(os.path.dirname(path))
    #    makeplot(path, path, cname)
    
    
    
