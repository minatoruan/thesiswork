import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from bpati import signals
from libs.arff import Dataset
from sklearn import cross_validation, datasets
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve

default_range_C = np.linspace(0.01, 20, 30)
default_range_variable = np.linspace(0.01, 20, 30)
coef0 = 1

svc_tuned_parameters = []
svc_tuned_parameters.append({'kernel': ['linear'], 'C': default_range_C})
svc_tuned_parameters.append({'kernel': ['poly'], 'degree': default_range_variable, 'C': default_range_C, 'coef0': [0.5, 1, 1.5]})
svc_tuned_parameters.append({'kernel': ['rbf'], 'gamma': default_range_variable, 'C': default_range_C, 'coef0': [0.5, 1, 1.5]})

def sensitivity_scores(cfm):
    sensitivity = (cfm[0][0]*1./(cfm[0][0] + cfm[0][1]))
    specificity = (cfm[1][1]*1./(cfm[1][1] + cfm[1][0]))
    precision = (cfm[0][0]*1./(cfm[0][0] + cfm[1][0]))
    ppv = (cfm[1][1]*1./(cfm[1][1] + cfm[0][1]))
    return sensitivity, specificity, precision, ppv

def scores(y_true, y_predict):
    cfm = confusion_matrix(y_true, y_predict)
    if cfm.shape == (2, 2): return sensitivity_scores(cfm)
    return None

def copy_SVC(estimator):
    return SVC(kernel=estimator.kernel, gamma=estimator.gamma, C=estimator.C, coef0=estimator.coef0, degree=estimator.degree)

def hamming_scoring(estimator, X, y):
    y_pred = estimator.predict(X)
    return hamming_loss(y, y_pred)

def predict_table(estimator, X, y):
    y_pred = estimator.predict(X)
    return [y, y_pred]

def plot_between(plt, x, scores, color, label):
    test_scores_mean = np.mean(scores, axis=1)
    test_scores_std = np.std(scores, axis=1)
    
    plt.fill_between(x, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color=color)
    plt.plot(x, test_scores_mean, 'o-', color=color, label=label)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure(figsize=(8,5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    plot_between(plt, train_sizes, train_scores, "r", "Training score")
    plot_between(plt, train_sizes, test_scores, "g", "Cross-validation score")
    
    plt.grid()
    plt.legend(loc="best")
    #plt.ylim((0,1.1))
    return plt

def plot_diagnosing_bias_variance(X, y, X_test = None, y_test = None, coef0=coef0, gammas=None, degrees=None, C=default_range_C[0], filename=''):
    cfms = []
    test_scores = []
    train_scores = []
    separate_test_score = []
    sensitivity = []
    specificity = []
    precision = []
    ppv = []
    SVCs = []
    
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    else:
        X_train, y_train = X, y
    
    plt.figure(0, figsize=(8,5))
    plt.figure(1, figsize=(8,5))
    if gammas != None:
        variables = gammas
        SVCs = [SVC(kernel='rbf', gamma=g, C=C, coef0=coef0) for g in gammas]
        plt.figure(0)
        plt.xlabel("Gamma")
        plt.figure(1)
        plt.xlabel("Gamma")
    else:
        variables = degrees
        SVCs = [SVC(kernel='poly', degree=d, C=C, coef0=coef0) for d in degrees]
        plt.figure(0)
        plt.xlabel("Degree")
        plt.figure(1)
        plt.xlabel("Degree")
    
    for clf in SVCs:
        test_scores.append(cross_validation.cross_val_score(copy_SVC(clf), X_train, y_train, cv=5, n_jobs = 4, scoring=hamming_scoring))
                    
        model = copy_SVC(clf).fit(X_train, y_train)
        train_scores.append(hamming_loss(y_train, model.predict(X_train)))
        separate_test_score.append(hamming_loss(y_test, model.predict(X_test)))
        scrs = scores(y_test, model.predict(X_test))
        print clf
        print confusion_matrix(y_test, model.predict(X_test))
        print 'Hamming loss', hamming_loss(y_test, model.predict(X_test))
        
        if scores is not None:
            sensitivity.append(scrs[0])
            specificity.append(scrs[2])
            precision.append(scrs[1])
            ppv.append(scrs[3])
            print 'Sensitivity', scrs[0], 
            print 'Specificity', scrs[2],
            print 'Precision', scrs[1],
            print 'PPV', scrs[3]
    
    plt.figure(0)
    plot_between(plt, variables, test_scores, "g", "Cross-validation score")
    plt.plot(variables, train_scores, 'o-', color="r", label="Training score")
    plt.plot(variables, separate_test_score, 'o-', color="b", label="Testing score")
    plt.ylabel("Score")
    plt.grid()
    plt.legend(loc="best")
    
    if scores is not None: 
        plt.figure(1)
        plt.plot(variables, sensitivity, 'o-', color="r", label="Sensitivity score")
        plt.plot(variables, specificity, 'o-', color="b", label="Specificity score")
        plt.plot(variables, precision, 'o-', color="g", label="Precision score")
        plt.plot(variables, ppv, 'o-', color="black", label="PPV score")    
        plt.ylabel("Score")
        plt.grid()
        plt.legend(loc="best")
    
    if filename == '': plt.show()   
    else:
        plt.figure(0)
        plt.savefig('%s_hammingloss.png' % filename)
        if scores is not None:
            plt.figure(1)
            plt.savefig('%s_sensitivity.png' % filename)
    return SVCs[np.array(separate_test_score).argmin()]

def plot_diagnosing_bias_variance_C(X, y, clf, X_test = None, y_test = None, C = default_range_C):
    test_scores = []
    train_scores = []
    separate_test_score = []
    
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    else:
        X_train, y_train = X, y
        
    for c in C:
        clf.C = c
        test_scores.append(cross_validation.cross_val_score(copy_SVC(clf), X, y, cv=5, n_jobs = 4, scoring = hamming_scoring))
        
        model = copy_SVC(clf).fit(X_train, y_train)
        train_scores.append(hamming_loss(y_train, model.predict(X_train)))        
        separate_test_score.append(hamming_loss(y_test, model.predict(X_test)))
           
    plot_between(plt, C, test_scores, "g", "Cross-validation score")
    plt.plot(C, train_scores, 'o-', color="r", label="Training score")
    plt.plot(C, separate_test_score, 'o-', color="b", label="Testing score")
        
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.title(clf)
    plt.grid()
    plt.legend(loc="best")
    #plt.ylim((0,1.1))
    plt.show()
    
def plot_diagnosing_bias_variance_SVC_poly(X, y, X_test = None, y_test = None, coef0=coef0, degrees = default_range_variable, C = default_range_C):
    clf_best = plot_diagnosing_bias_variance(X, y, X_test = X_test, y_test = y_test, coef0=coef0, degrees = degrees)
    plot_diagnosing_bias_variance_C(X, y, clf_best, C)
    
def plot_diagnosing_bias_variance_SVC_rbf(X, y, gammas = default_range_variable, C = default_range_C):
    clf_best = plot_diagnosing_bias_variance(X, y, gammas = gammas)
    plot_diagnosing_bias_variance_C(X, y, clf_best, C)