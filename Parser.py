#!/usr/bin/python

import argparse


def parse_command_line():

    usage = """for more details, visit http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    
    """
    
    parser = argparse.ArgumentParser(description=usage)

    #----- Load Data -----#
    parser.add_argument("--sqlite-file",action="store",default=None,type=str)
    parser.add_argument("--table-name",action="store",default=None,type=str)
    parser.add_argument("--feature-file",action="store",default=None,type=str)
    parser.add_argument("--class-criterion",action="store",default=None,type=str)
    parser.add_argument("--class-cut",action="store",default=None,type=float)

    parser.add_argument("--signal-txt-file",action="store",default=None,type=str)
    parser.add_argument("--background-txt-file",action="store",default=None,type=str)
    parser.add_argument("--binary-system",action="store",type=str)
    
    parser.add_argument("--training-size",action="store",type=float,default=0.5)
    parser.add_argument("--scaler",action="store",type=str,default=None)
    parser.add_argument("--output-dir",action="store",type=str)

    subparsers = parser.add_subparsers(dest='mla_selection')
    
    #----- Training Parameters for RF -----#
    group_rf = subparsers.add_parser("rf")
    group_rf.add_argument("--pkl-file",action="store",default=None,type=str)
    group_rf.add_argument("--n-estimators",action="store",type=int,default=100,\
                          help="default=100")
    group_rf.add_argument("--criterion",action="store",type=str,default="gini",\
                          help="default=gini, available options: gini or entropy")
    group_rf.add_argument("--max-features",action="store",default="auto",\
                          help="default=auto, If int, then consider max_features features at each split. If float between 0. and 1., then max_features is a persentage and int(max_features * n_features) features are considered at each split")
    group_rf.add_argument("--max-depth",default=None,\
                          help="default=None, available values: integer or None")
    group_rf.add_argument("--max-leaf-nodes",default=None,\
                          help="default=None, available values: integer or None")
    group_rf.add_argument("--min-samples-split",default=2,type=int)
    group_rf.add_argument("--min-samples-leaf",default=1,type=int)
    group_rf.add_argument("--n-jobs",action="store",type=int,default=1)
    #----- Diagnostic for RF -----#
    group_rf.add_argument("--estimators",default=False,action="store_true",help="list of DecisionTreeClassifier")
    group_rf.add_argument("--classes",default=False,action="store_true")
    group_rf.add_argument("--feature-importance",default=False,action="store_true")

    #----- Training Parameters for SVM -----#
    group_svm = subparsers.add_parser("svm")
    group_svm.add_argument("--pkl-file",action="store",default=None,type=str)
    group_svm.add_argument("--C",action="store",default=1.,type=float)
    group_svm.add_argument("--kernel",action="store",default="rbf",type=str,\
                           help="Options: linear, poly, rbf, sigmoid, precomputed")
    group_svm.add_argument("--degree",action="store",default=3,type=int,\
                           help="Degree of the polynomial kernel function. Ignored by all other kernels.")
    group_svm.add_argument("--gamma",action="store",default=0.,type=float,\
                           help="Kernel coefficient for rbf, poly and sigmoid. If gamma is 0. then 1/n_features will be used instead.")
    group_svm.add_argument("--coef0",action="store",default=0.,type=float,\
                           help="Independent term in kernel function. It is only significant in poly and sigmoid.")
    group_svm.add_argument("--probability",action="store_true",default=True)
    group_svm.add_argument("--tol",action="store",default=1e-3,type=float)
    group_svm.add_argument("--max-iter",action="store",default=10000,type=int,\
                           help="Hard limit on iterations within solver, or -1 for no limit.")
    group_svm.add_argument("--random-state",action="store",default=None)

    #----- Training Parameters for NN -----#
    group_nn = subparsers.add_parser("nn")
    group_nn.add_argument("--pkl-file",action="store",default=None,type=str)
    group_nn.add_argument("--hidden-layer-sizes",action="store",default=None,type=str)
    group_nn.add_argument("--activation",action="store",default="relu",type=str,\
                          help="default: relu; identity - no-op activation, returnes f(x)=x, logistic - logistic sigmoid function, returns f(x)=1/(1+exp(-x)), tanh - hyperbolic tan function, returns f(x)=tanh(x), or relu - rectified linear unit, returns f(x) = max(0,x)")
    group_nn.add_argument("--solver",action="store",default="adam",type=str,\
                          help="default: adam; lbgfs - optimizer in the family of quasi-Newton methods, sgd - stochastic gradient descent, or adam - stochastic gradient-based optimizer")
    group_nn.add_argument("--learning-rate",action="store",default="constant",\
                          type=str,\
                          help="default: constant; constant, invscaling, or adaptive")
    group_nn.add_argument("--max-iter",action="store",default=200,type=int)
    #----- Diagnostic for NN -----#
    group_nn.add_argument("--iterations",default=False,action="store_true")
    group_nn.add_argument("--classes",default=False,action="store_true")
    
    args = vars(parser.parse_args())
    
    return args
