# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

# scikit-learn libraries
from sklearn.svm import SVC
from sklearn import metrics

# utilities
import util

# csv
import csv

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

class MulticlassSVM :
    
    def __init__(self, R, C=1.0, kernel='linear', **kwargs) :
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
        self.svms = [None for _ in xrange(num_classifiers)]
        for j in xrange(num_classifiers) :
            svm = SVC(kernel=kernel, C=C[j], **kwargs)
            self.svms[j] = svm
    
    
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
            ### ========== TODO : START ========== ###
            # part b: setup training data for binary classification task
                        
            # HERE IS ONE WAY (THERE MAY BE OTHER APPROACHES)
            #
            # keep two lists, pos_ndx and neg_ndx, that store indices
            #   of examples to classify as pos / neg for current binary task
            #
            # for each class C
            # a) find indices for which examples have class equal to C
            #    [use np.nonzero(CONDITION)[0]]
            # b) update pos_ndx and neg_ndx based on output code R[i,j]
            #    where i = class index, j = classifier index
            #
            # set X_train using X with pos_ndx and neg_ndx
            # set y_train using y with pos_ndx and neg_ndx
            #   y_train should contain only {+1,-1}

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
            ### ========== TODO : END ========== ###
            
            # train binary classifier
            svm = self.svms[j]
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
            discrim_func[:,j] = self.svms[j].decision_function(X)
        
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
# main
######################################################################

def main() :

    original_labels = open('trainLabels_modified.csv', 'rb')
    labelreader = csv.reader(original_labels)

    classes = ['frog', 'deer', 'ship', 'airplane']
    train_y = np.zeros(1000)
    test_y = np.zeros(400)
    i = 0
    for row in labelreader:
        if i > 1400:
            break
        elif i > 1000:
            test_y[i - 1001] = classes.index(row[1])
        elif i > 0:
            train_y[i - 1] = classes.index(row[1])
        i += 1

    train_X = np.zeros((1000, 3072))
    test_X = np.zeros((400, 3072))
    for index in xrange(1400):
        name = "training_data/" + str(index+1) + ".png"
        img = mpimg.imread(name)
        if index < 1000:
            train_X[index] = img.flatten()
        else:
            test_X[index - 1000] = img.flatten()
        # plt.imshow(img)
        # plt.show()

    # average_pic = np.average(X, axis=0)
    # util.show_image(average_pic)

    num_classes = 4

    R_ovr = generate_output_codes(num_classes, 'ovr')

    # create MulticlassSVM
    # use SVMs with polynomial kernel of degree 4 : K(u,v) = (1 + <u,v>)^4
    # and slack penalty C = 10
    clf = MulticlassSVM(R_ovr, C=10, kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    err = metrics.zero_one_loss(test_y, y_pred, normalize=True)
    print 'ovr accuracy', 1 - err

    R_ovo = generate_output_codes(num_classes, 'ovo')

    clf = MulticlassSVM(R_ovo, C=10, kernel='poly', degree=2, gamma=1.0, coef0=1.0)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    err = metrics.zero_one_loss(test_y, y_pred, normalize=True)
    print 'ovo accuracy', 1 - err


    U, mu = util.PCA(train_X)
    #util.plot_gallery([util.vec_to_image(U[:,i]) for i in xrange(20)])

    l_list = [1, 10, 50, 100, 500, 1288]
    for l in l_list:
        Z, Ul = util.apply_PCA_from_Eig(train_X, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        # util.plot_gallery([X_rec[i, :] for i in xrange(12)])


if __name__ == "__main__" :
   main()