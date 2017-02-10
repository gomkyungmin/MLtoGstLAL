import os
from os.path import dirname,join,isdir,exists,split

import numpy as np
import sqlite3

from sklearn import preprocessing

import Utils as utils
import Plot as plt


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
    exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def LoadData(**args):

    if args['sqlite_file'] is not None:
        samples,outdir = LoadSQL(**args)
        return samples,outdir
    elif args['signal_txt_file'] is not None:
        samples,outdir = LoadTXT(**args)
        return samples,outdir

    
def LoadSQL(**args):

    sqlite_file = args['sqlite_file']
    print "\nData is loaded from %s" % sqlite_file

    module_path = dirname(__file__)
    base_dir = module_path

    output_path, filename = split(sqlite_file)
    outdir = output_path.replace('data',args['output_dir'])
    subdir = filename.replace(".sqlite","")

    outdir = join(outdir,subdir)
    if not os.path.exists(outdir):
        os.system('mkdir -p %s' % outdir)
    else:
        pass
    
    # Read feature variables
    features = np.genfromtxt(args['feature_file'],delimiter=',',dtype=str)
    print "\nfeatures:"
    print features
    
    conn = sqlite3.connect(sqlite_file)
    conn.row_factory = lambda cursor, row: row[0]
    cursor  = conn.cursor()    

    # Preapre flat data of N-d array with chosen feature variables
    for j,feature in enumerate(features):
        cursor.execute("SELECT {cn} from {tn}".format(cn=feature,tn=args['table_name']))
        vals = cursor.fetchall()
        vals = np.array(vals)
        if j == 0:
            flat_data = vals.reshape((len(vals),1))
        else:
            flat_data = np.append(flat_data,vals.reshape((len(vals),1)),1)

    print "Total # of samples: %d" % len(flat_data)

    # Decide class of each sample based on the chosen criterion and its value
    idx_cls_criterion = int(np.where(features==args['class_criterion'])[0][0])
    ranks = flat_data[:,idx_cls_criterion] >= args['class_cut']
    print "\n%s for class-cut: %f" % (args['class_criterion'],args['class_cut'])
    print "# of Foreground samples: %d, # of Background samples: %d" %\
        (np.sum(ranks), len(ranks)-np.sum(ranks))
    target = ranks.astype(np.int)

    # Combine flat data and class into (N+1)-d array and save it for reference    
    full_data = np.append(flat_data,target.reshape((len(target),1)),1)

    dat_origin="origin: %s" % args['sqlite_file']
    dat_header="%s\nmass1,mass2,snr,chisq,class" % dat_origin
    dat_fmt = "%f %f %f %f %f %f %d"
    np.savetxt(join(outdir,"full_data.dat"),\
               full_data,fmt=dat_fmt,header=dat_header)

    # Generate a collected scatter plot of all samples w/ all features
    plt.scatter_plot(flat_data,target,features,outdir,'full')
    
    return Bunch(features=features,\
                 data=flat_data,\
                 target=target),outdir


def LoadTXT(**args):

    txt_file_s = args['signal_txt_file']
    txt_file_b = args['background_txt_file']
    print "\nData is loaded from %s (for signal) and %s (for background)"\
        % (txt_file_s,txt_file_b)

    module_path = dirname(__file__)
    base_dir = module_path

    output_path, filename_s = split(txt_file_s)
    outdir = output_path.replace('data',args['output_dir'])
    subdir = filename_s.replace(".txt","")

    outdir = join(outdir,subdir)
    if not os.path.exists(outdir):
        os.system('mkdir -p %s' % outdir)
    else:
        pass
    
    # Read feature variables
    features = np.genfromtxt(args['feature_file'],delimiter=',',dtype=str)
    print "\nfeatures:"
    print features

    features = features.tolist()

    # Preapre flat data of N-d array with chosen feature variables
    get_column_name = np.genfromtxt(txt_file_s,delimiter='',\
                                     dtype=None,names=True)

    idx_feature = []
    for i,col in enumerate(get_column_name.dtype.names):
        for fea in features:
            if col == fea:
                idx_feature.append(i)
            else:
                pass

    data_s = np.genfromtxt(txt_file_s,delimiter='',\
                           dtype=None,skip_header=1,\
                           usecols=idx_feature)
    data_b = np.genfromtxt(txt_file_b,delimiter='',\
                           dtype=None,skip_header=1,\
                           usecols=idx_feature)

    flat_data = np.append(data_s,data_b,0)

    print "Total # of samples: %d" % len(flat_data)

    # Decide class of each sample based on the source of data either signal or background
    cls_s = np.ones(len(data_s))
    cls_b = np.zeros(len(data_b))
    target = np.append(cls_s,cls_b,0)

    # Combine flat data and class into (N+1)-d array and save it for reference
    assert len(flat_data) == len(target)
    full_data = np.append(flat_data,target.reshape(len(target),1),1)

    dat_origin="origin: %s / %s" % (args['signal_txt_file'],args['background_txt_file'])
    dat_header="%s\nmass1,mass2,spin1z,spin2z,snr,chisq,class" % dat_origin
    dat_fmt = "%f %f %f %f %f %f %d"
    np.savetxt(join(outdir,"full_data.dat"),\
               full_data,fmt=dat_fmt,header=dat_header)

    # Generate a collected scatter plot of all samples w/ all features
    plt.scatter_plot(flat_data,target,features,outdir,'full')
    
    return Bunch(features=features,\
                 data=flat_data,\
                 target=target),outdir


def LoadDAT(outdir,filenames):

    train_data = np.genfromtxt(join(outdir,filenames[0]),\
                               skip_header=2)
    test_data = np.genfromtxt(join(outdir,filenames[1]),\
                              skip_header=2)
    y_train = train_data[:,-1]
    X_train = np.delete(train_data,-1,1)
    y_test = test_data[:,-1]
    X_test = np.delete(test_data,-1,1)

    dataset = Bunch(X_train=X_train,\
                    X_test=X_test,\
                    y_train=y_train,\
                    y_test=y_test)

    return dataset


def DataScaler(X_train, X_test, scaler):

    if scaler == 'Standard':
        scaler = preprocessing.StandardScaler()
    elif scaler == 'MinMax':
        scaler = preprocessing.MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
                            

def DataGeneration(samples,outdir,filenames,**args):
        
    X = samples.data
    y = samples.target

    # Permuting the order of original data
    # in order to avoid biased data configuration
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    pX = X[p]
    py = y[p]

    train_samples = int(len(samples.data)*args['training_size'])

    X_train = pX[:train_samples]
    X_test = pX[train_samples:]
    y_train = py[:train_samples]
    y_test = py[train_samples:]

    dataset = Bunch(X_train=X_train,\
                    X_test=X_test,\
                    y_train=y_train,\
                    y_test=y_test)
    
    # Preprocessing (Scaling) for X_train and X_test
    if args['scaler'] is not None:
        if 'param_search_file' in args:
            pass
        else:
            print "\nA scaler, %s, is applied in data generation." % args['scaler']
        dataset.X_train, dataset.X_test\
            = DataScaler(dataset.X_train, dataset.X_test, args['scaler'])
    else:
        if 'param_search_file' in args:
            pass
        else:
            print "\nNo scaler is applied in data generation."

    # Generate a collected scatter plot of all samples w/ all features
    features = np.genfromtxt(args['feature_file'],delimiter=',',dtype=str)
    plt.scatter_plot(dataset.X_train,dataset.y_train,\
                     features,outdir,'train_'+str(args['scaler']))
    plt.scatter_plot(dataset.X_test,dataset.y_test,\
                     features,outdir,'test_'+str(args['scaler']))

    # Save generated train and test data into files for reference
    train_data = np.append(X_train,y_train.reshape((len(y_train),1)),1)
    test_data = np.append(X_test,y_test.reshape((len(y_test),1)),1)
    print "\nIn the generated data..."
    print "# of train samples: %d (%d signal/%d background)" %\
        (len(train_data),np.count_nonzero(y_train==1),np.count_nonzero(y_train==0))
    print "# of test samples: %d (%d signal/%d background)" %\
        (len(test_data),np.count_nonzero(y_test==1),np.count_nonzero(y_test==0))
    
    train_data_header\
        ="train_data (total:%d/signal:%d/background:%d)\nmass1,mass2,spin1z,spin2,snr,chisq,class" %\
        (len(train_data),np.count_nonzero(y_train==1),np.count_nonzero(y_train==0))
    test_data_header\
        ="test_data (total:%d/signal:%d/background:%d)\nmass1,mass2,spin1z,spin2z,snr,chisq,class" %\
        (len(test_data),np.count_nonzero(y_test==1),np.count_nonzero(y_test==0))
    dat_fmt = "%f %f %f %f %f %f %d"

    np.savetxt(join(outdir,filenames[0]),\
               train_data,fmt=dat_fmt,header=train_data_header)
    np.savetxt(join(outdir,filenames[1]),\
               test_data,fmt=dat_fmt,header=test_data_header)
    
    return dataset


def GetData(samples,outdir,**args):

    file_train_data = 'train_data_%s.dat' % str(args['scaler'])
    file_test_data = 'test_data_%s.dat' % str(args['scaler'])
    dat_filename = [file_train_data,file_test_data]
    
    if os.path.exists(join(outdir,dat_filename[0])) and\
       os.path.exists(join(outdir,dat_filename[1])):
        print "\nYou already have train and test data."
        answer = utils.QueryYesNo("Do you want to generate new train and test data?")
        if answer == True:
            dataset = DataGeneration(samples,outdir,dat_filename,**args)
        elif answer == False:
            dataset = LoadDAT(outdir,dat_filename)
    else:
        dataset = DataGeneration(samples,outdir,dat_filename,**args)

    return dataset
