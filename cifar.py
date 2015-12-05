# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

# pyleargist libraries
from PIL import Image
import leargist

# scikit-learn libraries
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

#scikit-image libraries
import skimage.feature as ft

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

# OpenCV libraries
import cv2

# utilities
import util

import csv

import os.path

cifar_imageSize = (32,32)


######################################################################
# output code functions
######################################################################

def generate_output_codes(num_classes, code_type,
                          num_classifiers=None, num_repeats=10000) :
    """
    Generate output codes for multiclass classification.
    
    Parameters
    --------------------
        num_classes     -- int, number of classes
        code_type       -- string, type of output code
                           allowable: 'ovr', 'ovo', 'rand'
        num_classifiers -- int, number of classifiers
                           (used only if code_type == 'rand')
        num_repeats     -- int, number of output codes from which to select
                           (used only if code_type == 'rand')
    
    Returns
    --------------------
        R               -- numpy array of shape (num_classes, num_classifiers),
                           output code
    """
    
    if code_type == "ovr" :      # one vs rest (one vs all)
        R = -1*np.ones((num_classes, num_classes))
        for t in xrange(num_classes) :
            R[t,t] = 1
    
    elif code_type == 'ovo' :    # one vs one (all-pairs)
        num_classifiers = num_classes * (num_classes-1)/2
        R = np.zeros((num_classes, num_classifiers))
        t = 0
        for i in xrange(num_classes) :
            for j in xrange(i+1, num_classes) :
                R[i,t] = 1
                R[j,t] = -1
                t += 1
    else :
        raise Exception("Error! Unknown code type!")
    
    return R


######################################################################
# loss functions
######################################################################

def compute_losses(loss_type, R, discrim_func, alpha=2) :
    """
    Given output code and distances (for each example), compute losses (for each class).
    
    hamming  : Loss  = (1 - sign(z)) / 2
    sigmoid  : Loss = 1 / (1 + exp(alpha * z))
    logistic : Loss = log(1 + exp(-alpha * z))
    
    Parameters
    --------------------
        loss_type    -- string, loss function
                        allowable: 'hamming', 'sigmoid', 'logistic'
        R            -- numpy array of shape (num_classes, num_classifiers)
                        output code
        discrim_func -- numpy array of shape (num_classifiers,)
                        distance of samples to hyperplane, one per example
        alpha        -- float, parameter for sigmoid and logistic functions
    
    Returns
    --------------------
        losses       -- numpy array of shape (num_classes,), losses
    """
    
    # element-wise multiplication of matrices of shape (num_classes, num_classifiers)
    # tiled matrix created from (vertically) repeating discrim_func num_classes times
    z = R * np.tile(discrim_func, (R.shape[0],1))    # element-wise
    
    # compute losses in matrix form
    if loss_type == 'hamming' :
        losses = np.abs(1 - np.sign(z)) * 0.5
    
    elif loss_type == 'sigmoid' :
        losses = 1./(1 + np.exp(alpha * z))
    
    elif loss_type == 'logistic' :
        # compute in this way to avoid numerical issues
        # log(1 + exp(-alpha * z)) = -log(1 / (1 + exp(-alpha * z)))
        eps = np.spacing(1) # numpy spacing(1) = matlab eps
        val = 1./(1 + np.exp(-alpha * z))
        losses = -np.log(val + eps)
    
    else :
        raise Exception("Error! Unknown loss function!")
    
    # sum over losses of binary classifiers to determine loss for each class
    losses = np.sum(losses, 1) # sum over each row
    
    return losses

def logistic_losses(R, discrim_func, alpha=2) :
    """
    Wrapper around compute_losses for logistic loss function.
    """
    return compute_losses('logistic', R, discrim_func, alpha)


######################################################################
# classes
######################################################################

class Multiclass :
    
    def __init__(self, R, clf, C=1.0, kernel='linear', **kwargs) :
        """
        Multiclass SVM.
        
        Attributes
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            svms    -- list of length num_classifiers
                       binary classifiers, one for each column of R
            classes -- numpy array of shape (num_classes,) classes
        
        Parameters
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            C       -- numpy array of shape (num_classifiers,1) or float
                       penalty parameter C of the error term
            kernel  -- string, kernel type
                       see SVC documentation
            kwargs  -- additional named arguments to SVC
        """
        
        num_classes, num_classifiers = R.shape
        
        # store output code
        self.R = R
        
        # use first value of C if dimension mismatch
        try :
            if len(C) != num_classifiers :
                raise Warning("dimension mismatch between R and C " +
                                "==> using first value in C")
                C = np.ones((num_classifiers,)) * C[0]
        except :
            C = np.ones((num_classifiers,)) * C
        
        # set up and store classifier corresponding to jth column of R
        self.clfs = [None for _ in xrange(num_classifiers)]
        for j in xrange(num_classifiers) :
            if clf == "svm":
                clfs = SVC(kernel=kernel, C=C[j], **kwargs)
            elif clf == "logistic":
                clfs = LogisticRegression(fit_intercept=True, C=C[j])
            self.clfs[j] = clfs
    
    
    def fit(self, X, y) :
        """
        Learn the multiclass classifier (based on SVMs).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), features
            y    -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        classes = np.unique(y)
        num_classes, num_classifiers = self.R.shape
        if len(classes) != num_classes :
            raise Exception('num_classes mismatched between R and data')
        self.classes = classes    # keep track for prediction
        
        # iterate through binary classifiers
        for j in xrange(num_classifiers) :

            pos_ndx = []
            neg_ndx = []
            R = self.R

            for i in xrange(num_classes):
                indices = np.nonzero(y == classes[i])[0].tolist()
                if R[i][j] == 1:
                    pos_ndx += indices
                if R[i][j] == -1:
                    neg_ndx += indices
            
            X_train = X[pos_ndx + neg_ndx, :]
            y_train = np.append(np.ones(len(pos_ndx)), np.ones(len(neg_ndx)) * (-1))
            
            # train binary classifier
            svm = self.clfs[j]
            svm.fit(X_train, y_train)
    
    
    def predict(self, X, loss_func=logistic_losses) :
        """
        Predict the optimal class.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            loss_func -- loss function
                         allowable: hamming_losses, logistic_losses, sigmoid_losses
        
        Returns
        --------------------
            y         -- numpy array of shape (n,), predictions
        """
        
        n,d = X.shape
        num_classes, num_classifiers = self.R.shape
        
        # setup predictions
        y = np.zeros(n)
        
        # discrim_func is a matrix that stores the discriminant function values
        #   row index represents the index of the data point
        #   column index represents the index of binary classifiers
        discrim_func = np.zeros((n,num_classifiers))
        for j in xrange(num_classifiers) :
            discrim_func[:,j] = self.clfs[j].decision_function(X)
        
        # scan through the examples
        losses = []
        for i in xrange(n) :
            # compute losses of each class
            losses = loss_func(self.R, discrim_func[i,:])
            
            # predict the label as the one with the minimum loss
            ndx = np.argmin(losses)
            y[i] = self.classes[ndx]
        
        return y


######################################################################
# functions -- evaluation
######################################################################

def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    # compute average cross-validation performance
    index = 0
    scores = np.zeros(10) 
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores[index] = metrics.accuracy_score(y_test, y_pred)
        index += 1
        # print "Fold", index, ":", scores[index - 1]
    return (sum(scores)) / 10


def select_param_randomForest(X, y, kf):

    print 'Random Forest Hyperparameter Selection:'
    ### ========== TODO : START ========== ###
    numTree_range = [10, 20, 50, 100, 200, 500]
    depth_range = [10, 20, 50, 100, 200, 500, 1000, 2000]
    best_score = float("-inf")
    best_numTree = 0
    best_depth = 0
    for numTree in numTree_range:
        for depth in depth_range:
            clf = RandomForestClassifier(n_estimators=numTree, max_depth = depth, criterion='entropy')
            temp_score = cv_performance(clf, X, y, kf)
            print "The accuracy for numTree =", numTree , "and max depth =", depth, "is", temp_score
            if temp_score > best_score:
                best_numTree, best_depth, best_score  = numTree, depth, temp_score
    return best_numTree, best_depth


def select_param_randomForest(X, y, kf):
    
    print 'Random Forest Hyperparameter Selection:'
    ### ========== TODO : START ========== ###
    numTree_range = [10, 20, 50, 100, 200, 500]
    depth_range = [10, 20, 50, 100, 200, 500, 1000, 2000]
    best_score = float("-inf")
    best_numTree = 0
    best_depth = 0
    for numTree in numTree_range:
        for depth in depth_range:
            clf = RandomForestClassifier(n_estimators=numTree, max_depth = depth, criterion='entropy')
            temp_score = cv_performance(clf, X, y, kf)
            print "The accuracy for numTree =", numTree , "and max depth =", depth, "is", temp_score
            if temp_score > best_score:
                best_numTree, best_depth, best_score  = numTree, depth, temp_score
    return best_numTree, best_depths


######################################################################
# main
######################################################################

def main() :

    np.random.seed(1234)

    original_labels = open('trainLabels_modified.csv', 'rb')
    labelreader = csv.reader(original_labels)

    classes = ['frog', 'deer', 'ship', 'airplane']
    all_y = np.zeros(20000)
    i = 0
    for row in labelreader:
        if i > 0:
            all_y[i - 1] = classes.index(row[1])
        i += 1

    # train_y = np.zeros(3000)
    # valid_y = np.zeros(1000)

    train_X_raw = np.zeros((4000, 3072))
    train_y = np.zeros(4000)

    i = 0
    for index in xrange(20000):
        name = "data4000/" + str(index + 1) + ".png"
        if os.path.isfile(name):
            img = mpimg.imread(name)
            train_X_raw[i] = img.flatten()
            train_y[i] = all_y[index]
            i += 1

    # create stratified folds (10-fold CV)
    kf = StratifiedKFold(train_y, n_folds=10)


    numTree, depth = select_param_randomForest(train_X_raw, train_y, kf)

    clf = RandomForestClassifier(n_estimators=numTree, max_depth=depth, criterion='entropy')
    # clf.fit(train_X_raw, train_y)
    # y_pred = clf.predict(valid_X_raw)
    # err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    accuracy = cv_performance(clf, train_X_raw, train_y, kf)
    print 
    print '     Random forest with %d trees, each with max depth %d accuracy %f'  % (numTree, depth, accuracy)

    k = 14
    clf = KNeighborsClassifier(n_neighbors=k)
    #clf.fit(train_X_raw, train_y)
    accuracy = cv_performance(clf, train_X_raw, train_y, kf)
    print '     KNN with %d neighbors accuracy %f' % (k, accuracy)

    exit(0)

    # select_param_svm_poly(train_X_raw, train_y, kf)

    # Raw Feature

    # train_X_raw = np.zeros((3000, 3072))
    # valid_X_raw = np.zeros((1000, 3072))

    # i = 0
    # for index in xrange(20000):
    #     name = "training_data/" + str(index + 1) + ".png"
    #     if os.path.isfile(name):
    #         img = mpimg.imread(name)
    #         train_X_raw[i] = img.flatten()
    #         train_y[i] = all_y[index]
    #         i += 1

    # i = 0
    # for index in xrange(20000):
    #     name = "held_out/" + str(index + 1) + ".png"
    #     if os.path.isfile(name):
    #         img = mpimg.imread(name)
    #         valid_X_raw[i] = img.flatten()
    #         valid_y[i] = all_y[index]
    #         i += 1

    print "Done with loading data..."

    num_classes = 4

    R_ovr = generate_output_codes(num_classes, 'ovr')
    R_ovo = generate_output_codes(num_classes, 'ovo')

    # create MulticlassSVM
    # use SVMs with polynomial kernel of degree 2 : K(u,v) = (1 + <u,v>)^2
    # and slack penalty C = 10
    print "Raw feature:"
    clf = Multiclass(R_ovr, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_raw, train_y)
    y_pred = clf.predict(valid_X_raw)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_raw, train_y)
    y_pred = clf.predict(valid_X_raw)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM ovo accuracy', 1 - err

    clf = Multiclass(R_ovr, C=10, clf='logistic' )
    clf.fit(train_X_raw, train_y)
    y_pred = clf.predict(valid_X_raw)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='logistic')
    clf.fit(train_X_raw, train_y)
    y_pred = clf.predict(valid_X_raw)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg ovo accuracy', 1 - err

    clf = LogisticRegression(fit_intercept=True, C=10, penalty='l1', solver='lbfgs', multi_class='multinomial')
    clf.fit(train_X_raw, train_y)
    y_pred = clf.predict(valid_X_raw)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg multinomial accuracy', 1 - err


    # Extract Features using GIST Descriptor
    train_X_gist = np.zeros((3000, 960))
    valid_X_gist = np.zeros((1000, 960))
    i = 0
    for index in xrange(20000):
        name = "training_data/" + str(index + 1) + ".png"
        if os.path.isfile(name):
            img = Image.open(name)
            gist = leargist.color_gist(img)
            train_X_gist[i] = gist
            i += 1

    i = 0
    for index in xrange(20000):
        name = "held_out/" + str(index + 1) + ".png"
        if os.path.isfile(name):
            img = Image.open(name)
            gist = leargist.color_gist(img)
            valid_X_gist[i] = gist
            i += 1

    print "GIST (without PCA):"
    clf = Multiclass(R_ovr, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_gist, train_y)
    y_pred = clf.predict(valid_X_gist)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM GIST ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_gist, train_y)
    y_pred = clf.predict(valid_X_gist)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM GIST ovo accuracy', 1 - err

    clf = Multiclass(R_ovr, C=10, clf='logistic', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_gist, train_y)
    y_pred = clf.predict(valid_X_gist)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg GIST ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='logistic', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_gist, train_y)
    y_pred = clf.predict(valid_X_gist)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg GIST ovo accuracy', 1 - err

    clf = LogisticRegression(fit_intercept=True, C=10, penalty='l1', solver='lbfgs', multi_class='multinomial')
    clf.fit(train_X_gist, train_y)
    y_pred = clf.predict(valid_X_gist)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg GIST multinomial accuracy', 1 - err


   # Extract Features using HOG Descriptor
    train_X_hog = np.zeros((3000, 324))
    valid_X_hog = np.zeros((1000, 324))
    i = 0
    for index in xrange(20000):
        name = "training_data/" + str(index + 1) + ".png"
        if os.path.isfile(name):
            img = cv2.imread(name, 0)
            hog = ft.hog(img)
            train_X_hog[i] = hog
            i += 1

    i = 0
    for index in xrange(20000):
        name = "held_out/" + str(index + 1) + ".png"
        if os.path.isfile(name):
            img = cv2.imread(name, 0)
            hog = ft.hog(img)
            valid_X_hog[i] = hog
            i += 1

    print "HOG (without PCA):"
    clf = Multiclass(R_ovr, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_hog, train_y)
    y_pred = clf.predict(valid_X_hog)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM HOG ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_hog, train_y)
    y_pred = clf.predict(valid_X_hog)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM HOG ovo accuracy', 1 - err

    clf = Multiclass(R_ovr, C=10, clf='logistic', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_hog, train_y)
    y_pred = clf.predict(valid_X_hog)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg HOG ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='logistic', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_hog, train_y)
    y_pred = clf.predict(valid_X_hog)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg HOG ovo accuracy', 1 - err

    clf = LogisticRegression(fit_intercept=True, C=10, penalty='l1', solver='lbfgs', multi_class='multinomial')
    clf.fit(train_X_hog, train_y)
    y_pred = clf.predict(valid_X_hog)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg HOG multinomial accuracy', 1 - err


    # Combine raw features, GIST and HOG descriptors together as our new feature vectors
    train_X = np.concatenate((train_X_raw, train_X_gist, train_X_hog), axis=1)  # (3000, 4356)
    valid_X = np.concatenate((valid_X_raw, valid_X_gist, valid_X_hog), axis=1)  # (1000, 4356)

    print "Combine raw features, GIST and HOG descriptors together (without PCA):"
    clf = Multiclass(R_ovr, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(valid_X)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM HOG ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(valid_X)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM HOG ovo accuracy', 1 - err

    clf = Multiclass(R_ovr, C=10, clf='logistic', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(valid_X)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg HOG ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='logistic', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(valid_X)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg HOG ovo accuracy', 1 - err

    clf = LogisticRegression(fit_intercept=True, C=10, penalty='l1', solver='lbfgs', multi_class='multinomial')
    clf.fit(train_X, train_y)
    y_pred = clf.predict(valid_X)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg HOG multinomial accuracy', 1 - err


    # Using PCA on raw features
    l = 500
    print "PCA with %d principal components on raw features:" % l
    U_train, mu_train = util.PCA(train_X_raw)
    U_valid, mu_valid = util.PCA(valid_X_raw)
    Z_train, Ul_train = util.apply_PCA_from_Eig(train_X_raw, U_train, l, mu_train)
    train_X_rec = util.reconstruct_from_PCA(Z_train, Ul_train, mu_train)
    Z_valid, Ul_valid = util.apply_PCA_from_Eig(valid_X_raw, U_valid, l, mu_valid)
    valid_X_rec = util.reconstruct_from_PCA(Z_valid, Ul_valid, mu_valid)

    # create Multiclass
    # use SVMs with polynomial kernel of degree 2 : K(u,v) = (1 + <u,v>)^2
    # and slack penalty C = 10
    clf = Multiclass(R_ovr, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM PCA ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM PCA ovo accuracy', 1 - err

    clf = Multiclass(R_ovr, C=10, clf='logistic')
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg PCA ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='logistic')
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg PCA ovo accuracy', 1 - err

    clf = LogisticRegression(fit_intercept=True, C=10, penalty='l2', solver='lbfgs', multi_class='multinomial')
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg multinomial accuracy', 1 - err


    # Using PCA on combined features
    print "PCA with %d principal components on combined features:" % l
    U_train, mu_train = util.PCA(train_X)
    U_valid, mu_valid = util.PCA(valid_X)
    Z_train, Ul_train = util.apply_PCA_from_Eig(train_X, U_train, l, mu_train)
    train_X_rec = util.reconstruct_from_PCA(Z_train, Ul_train, mu_train)
    Z_valid, Ul_valid = util.apply_PCA_from_Eig(valid_X, U_valid, l, mu_valid)
    valid_X_rec = util.reconstruct_from_PCA(Z_valid, Ul_valid, mu_valid)

    # create Multiclass
    # use SVMs with polynomial kernel of degree 2 : K(u,v) = (1 + <u,v>)^2
    # and slack penalty C = 10
    clf = Multiclass(R_ovr, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM PCA ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='svm', kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     SVM PCA ovo accuracy', 1 - err

    clf = Multiclass(R_ovr, C=10, clf='logistic')
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg PCA ovr accuracy', 1 - err

    clf = Multiclass(R_ovo, C=10, clf='logistic')
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg PCA ovo accuracy', 1 - err

    clf = LogisticRegression(fit_intercept=True, C=10, penalty='l2', solver='lbfgs', multi_class='multinomial')
    clf.fit(train_X_rec, train_y)
    y_pred = clf.predict(valid_X_rec)
    err = metrics.zero_one_loss(valid_y, y_pred, normalize=True)
    print '     Log Reg multinomial accuracy', 1 - err



if __name__ == "__main__" :
   main()
