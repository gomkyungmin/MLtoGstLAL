#!/usr/bin/python

import os
from os.path import exists,join,isdir
import argparse

import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing

import Parser
import LoadData as ld
import Plot as plt
import MLA
import Utils as utils


def training_go_or_stop(dataset,pkldir,**args):

    tag = utils.Tag(**args)
    pklfilename = 'trained'+tag+'.pkl'
    pklfile = join(pkldir,pklfilename)
    
    if args['mla_selection'] == 'rf':
        mla_name = 'RF'
    elif args['mla_selection'] == 'svm':
        mla_name = 'SVM'
    elif args['mla_selection'] == 'nn':
        mla_name = 'NN'
        
    if os.path.exists(pklfile):
        print "\nYou already have a trained result which is trained with the same parameters!"
        answer = utils.QueryYesNo("Do you want to train %s again?" % mla_name)
        if answer == True:
            ml = MLA.MLA(tag,dataset,pkldir,**args)
            clf = ml.train()
        elif answer == False:
            ml = MLA.MLA(tag,dataset,pkldir,**args)
            clf = joblib.load(pklfile)
            print "Trained result is loaded from %s" % (pklfile)
            print clf
    else:
        ml = MLA.MLA(tag,dataset,pkldir,**args)
        clf = ml.train()

    return ml,clf


def evaluation(ml,clf,outdir):

    classified_result_proba = ml.test(clf)
    print classified_result_proba
    np.savetxt(join(outdir,"result_proba%s.txt") % ml.tag,classified_result_proba)
    
    return classified_result_proba


def performance_test(ml,clf,result,features,outdir,**args):

    plt.draw_roc(ml.y_test,result,ml.tag,outdir,args['mla_selection'])

    if args['mla_selection'] == 'rf' and\
       (args['estimators'] is True or\
        args['classes'] is True or\
        args['feature_importance'] is True):
        print("\n=== Diagnostic(s) of Trained RandomForest ===")
        MLA.DiagnosingTrainedForest(clf,features,**args)
    elif args['mla_selection'] == 'nn' and\
         (args['iterations'] is True or\
          args['classes'] is True):
        print("\n=== Diagnostic(s) of Trained Neural Network ===")
        MLA.DiagnosingTrainedNeuralNetwork(clf,**args)
    else:
        pass

    
def evaluation_go_or_stop(ml,clf,samples,outdir,**args):

    figfile = outdir+'/figure/plot_roc_lin'+ml.tag+'.png'
    
    if os.path.exists(figfile):
        print "\nYou already have results of the performance test. "
        answer = utils.QueryYesNo("Do you want to update previous results?")
        if answer == True:
            result = evaluation(ml,clf,outdir)
            performance_test(ml,clf,result,samples.features,outdir,**args)
        elif answer == False:
            pass
    else:
        result = evaluation(ml,clf,outdir)
        performance_test(ml,clf,result,samples.features,outdir,**args)

        
def main():

    args = Parser.parse_command_line()
    
    samples,outdir = ld.LoadData(**args)
    dataset = ld.GetData(samples,outdir,**args)    
    
    # Creation of a directory for trained pickle file
    pkldir = join(outdir,'trained_pickle')
    if not os.path.isdir(pkldir):
        os.system('mkdir -p %s' % pkldir)
    else:
        pass
    
    if args['pkl_file'] is not None:
        tag = args['pkl_file'].replace('%s/trained' % pkldir,'').replace('.pkl','')
        ml = MLA.MLA(tag, dataset, **args)
        clf = joblib.load(args['pkl_file'])
        print "\nTrained forest is loaded from %s" % (args['pkl_file'])
        print clf        
    else:
        ml,clf = training_go_or_stop(dataset,pkldir,**args)

    evaluation_go_or_stop(ml,clf,samples,outdir,**args)


if __name__=='__main__':

    main()
