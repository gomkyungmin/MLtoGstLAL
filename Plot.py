import os
from os.path import isdir,join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

from sklearn.metrics import roc_curve, auc


# font set
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('font', serif='Times New Roman')
plt.rc('font', size=20)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=15)


def draw_histogram(y_test,ranks,tag,figdir):

    idx_s = np.where(y_test == 1)[0]
    idx_b = np.where(y_test == 0)[0]

    ranks_s = ranks[idx_s]
    ranks_b = ranks[idx_b]

    num_bins = 80
    
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.grid(True,which='both')
    plt.hist(ranks_s, bins=num_bins, histtype='barstacked', color='b',\
             range=(0,1), label='signal samples',linestyle='solid',linewidth=2,\
             hatch='/', alpha=0.5)
    plt.hist(ranks_b, bins=num_bins, histtype='barstacked', color='r',\
             range=(0,1), label='background samples',linestyle='solid',linewidth=2,\
             hatch='\\', alpha=0.5)

    plt.legend(loc=1)
    plt.xlabel('Rank')
    plt.ylabel('Number of Classified Samples')
    plt.tight_layout()
    plt.savefig(figdir+'/plot_histogram_lin'+tag+'.png', fmt='png')

    print "%s is saved in %s." % ('plot_histogram_lin'+tag+'.png',figdir)
    
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.grid(True,which='both')
    plt.hist(ranks_s, bins=num_bins, histtype='barstacked', color='b',\
             range=(0,1), label='signal samples',linestyle='solid',linewidth=2,\
             hatch='/', alpha=0.5, log='True')
    plt.hist(ranks_b, bins=num_bins, histtype='barstacked', color='r',\
             range=(0,1), label='background samples',linestyle='solid',linewidth=2,\
             hatch='\\', alpha=0.5, log='True')

    plt.legend(loc=1)
    plt.xlabel('Rank')
    plt.ylabel('Number of Classified Samples')
    plt.ylim([0,len(ranks_b)])
    plt.tight_layout()
    plt.savefig(figdir+'/plot_histogram_ylog'+tag+'.png', fmt='png')

    print "%s is saved in %s." % ('plot_histogram_ylog'+tag+'.png',figdir)
    

def draw_roc(y_test,result_proba,tag,outdir,mla):

    figdir = join(outdir,'figure')
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    else:
        pass
    
    fpr, tpr, threshold = roc_curve(y_test,\
                                    result_proba[:,1],\
                                    drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    np.savetxt(join(outdir,"fpr%s.txt") % tag,fpr)
    np.savetxt(join(outdir,"tpr%s.txt") % tag,tpr)

    x_min = 1./len(y_test)

    random_guess_x = np.linspace(x_min,1,num=len(y_test),endpoint=True)
    random_guess_y = random_guess_x

    labels = {'rf':'Random Forest', 'svm':'Support Vector Machine', 'nn':'Neural Network'}
    label = labels[mla]
    
    print "\nPlotting figures..."

    plt.clf()
    plt.figure(figsize=(8,6))
    plt.grid(True,which='both')
    plt.xlim([0,1.01])
    plt.ylim([0,1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr,tpr,'-',label=label)
    plt.plot(random_guess_x,random_guess_y,'--',color='0.5',label='random guess')
    plt.text(0.2, 0.4, 'auc=%f' % roc_auc)
    plt.legend(loc='lower right')
    plt.savefig(figdir+'/plot_roc_lin'+tag+'.png', fmt='png')

    print "%s is saved in %s." % ('plot_roc_lin'+tag+'.png',figdir)
    
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.grid(True,which='both')
    plt.xlim([x_min,1.01])
    plt.ylim([0,1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.semilogx(fpr,tpr,'-',label=label)
    plt.semilogx(random_guess_x,random_guess_y,'--',color='0.5',label='random guess')
    plt.text(7e-4, 0.17, 'auc=%f' % roc_auc)
    plt.legend(loc='upper left')
    plt.savefig(figdir+'/plot_roc_xlog'+tag+'.png', fmt='png')

    print "%s is saved in %s." % ('plot_roc_xlog'+tag+'.png',figdir)

    draw_histogram(y_test,result_proba[:,1],tag,figdir)


def scatter_plot(data,target,features,outdir,data_type):

    figdir = join(outdir,'figure')
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    else:
        pass

    figname = 'plot_scatter_%s_data.png' % data_type
    if os.path.exists(join(figdir,figname)):
        print "\nYou have a %s already. Skipping plotting a scatter plot of %s data." % (figname,data_type)
        pass
    else:
        idx_s = np.where(target == 1)[0]
        idx_b = np.where(target == 0)[0]

        data_s = data[idx_s]
        data_b = data[idx_b]

        plt.clf()
        plt.figure(figsize=(17,9))

        i = 1
        for j in range(len(features)):

            j_max = len(features)-1
            data_x_s,data_y_s,data_x_b,data_y_b = 0,0,0,0
            xlabel,ylabel = '',''
            if j != j_max:
                data_x_s = data_s[:,j]
                data_y_s = data_s[:,j+1]
                data_x_b = data_b[:,j]
                data_y_b = data_b[:,j+1]
            
                xlabel = features[j]
                ylabel = features[j+1]
            elif j == j_max:
                data_x_s = data_s[:,j]
                data_y_s = data_s[:,0]
                data_x_b = data_b[:,j]
                data_y_b = data_b[:,0]
                xlabel = features[j]
                ylabel = features[0]
            
            x_min_s, x_max_s = data_x_s.min(), data_x_s.max()
            y_min_s, y_max_s = data_y_s.min(), data_y_s.max()
            x_min_b, x_max_b = data_x_b.min(), data_x_b.max()
            y_min_b, y_max_b = data_y_b.min(), data_y_b.max()

            x_min, x_max = min(x_min_s,x_min_b), max(x_max_s,x_max_b)
            y_min, y_max = min(y_min_s,y_min_b), max(y_max_s,y_max_b)

            h_x = (x_max - x_min)/5.
            h_y = (y_max - y_min)/5
            xx, yy = np.meshgrid(np.arange(x_min,x_max,h_x),\
                                 np.arange(y_min,y_max,h_y))

            ax = plt.subplot(2, len(features)/2., i)
            ax.scatter(data_x_b, data_y_b, edgecolors='k', color='r', alpha=0.5)
            ax.scatter(data_x_s, data_y_s, edgecolors='k', color='b', alpha=0.5)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            i += 1

        plt.tight_layout()
        plt.savefig(figdir+'/'+figname, fmt='png')
        print "\n%s is saved in %s." % (figname,figdir)            
            
